"""Measure T1-OCC target visibility in the compiled scene."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import mujoco
import numpy as np
from robosuite.models.objects import MujocoXMLObject

from ontimebench.scenarios.spec import (
    ContactPhysicsSpec,
    DrivenInterceptionGeometrySpec,
    FixtureStateSpec,
    Vector2,
    Vector3,
)

_SENSOR_FREQUENCY_HZ = 20
_SENSOR_PERIOD_S = 1.0 / _SENSOR_FREQUENCY_HZ
_TIMESTAMP_TOLERANCE_S = 1e-12
_INITIAL_VISIBLE_RATIO = 0.95
_INITIAL_VISIBLE_FRAMES = 8
_FULLY_HIDDEN_RATIO = 0.05
_MINIMUM_HIDDEN_FRAMES = 10
_MAXIMUM_HIDDEN_FRAMES = 60
_REAPPEARANCE_RATIO = 0.80
_REAPPEARANCE_FRAMES = 3
_MINIMUM_LIFT_ENVELOPE_CLEARANCE_M = 0.20

_VisibilityPhase: TypeAlias = Literal[
    "initial_visible",
    "await_hidden",
    "hidden",
    "await_reappearance",
    "reappearance",
    "complete",
    "failed",
]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


@dataclass(frozen=True, slots=True)
class _VisibilityFailureFacts:
    initial_visibility_lost: bool
    hidden_run_too_long: bool


@dataclass(frozen=True, slots=True)
class _OcclusionVisibilityState:
    frame_index: int
    sensor_time_s: float
    target_visible_ratio: float
    phase: _VisibilityPhase
    initial_visible_frames: int
    hidden_run_frames: int
    selected_hidden_duration_s: float | None
    reappearance_frames: int
    complete: bool
    failure_facts: _VisibilityFailureFacts
    failure_codes: tuple[str, ...]


_OccRow: TypeAlias = tuple[float, float, tuple[float, float, float], bool, bool]


def _occlusion_metrics(rows: Sequence[_OccRow]) -> dict[str, object]:
    """Return the source protocol's selected interval, metrics and failure codes."""
    if not rows:
        return {"complete": False, "failure_codes": ("trace_empty",), "metrics": {}}

    def runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
        found = []
        start = None
        for index, flag in enumerate((*flags, False)):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                found.append((start, index - 1))
                start = None
        return found

    def interval(run: tuple[int, int]) -> tuple[float, float, float, int]:
        start, end = run
        return (
            rows[start][0],
            rows[end][0] + 0.05,
            rows[end][0] + 0.05 - rows[start][0],
            end - start + 1,
        )

    failures: list[str] = []
    if any(rows[i][0] <= rows[i - 1][0] for i in range(1, len(rows))):
        failures.append("timestamps_not_strictly_increasing")
    initial_end = -1
    for index, row in enumerate(rows):
        if row[1] < 0.95:
            break
        initial_end = index
    initial_frames = initial_end + 1
    initial_duration = 0.0 if initial_end < 0 else interval((0, initial_end))[2]
    initial_displacement = 0.0 if initial_end < 0 else math.dist(rows[0][2], rows[initial_end][2])
    if initial_frames < 8:
        failures.append("initial_visible_frames_insufficient")
    if initial_duration + 1e-12 < 0.4:
        failures.append("initial_visible_duration_too_short")
    if initial_displacement + 1e-12 < 0.012:
        failures.append("initial_visible_displacement_too_small")
    hidden_runs = runs([row[1] <= 0.05 for row in rows])
    contacts = tuple(i for i, row in enumerate(rows) if row[3])
    holds = tuple(i for i, row in enumerate(rows) if row[4])

    def inside(run: tuple[int, int], indices: Sequence[int]) -> tuple[int, ...]:
        return tuple(i for i in indices if run[0] <= i <= run[1])

    def with_margin(run: tuple[int, int], indices: Sequence[int]) -> tuple[int, ...]:
        start, end, _, _ = interval(run)
        return tuple(
            i
            for i in inside(run, indices)
            if rows[i][0] - start + 1e-12 >= 0.1 and end - rows[i][0] + 1e-12 >= 0.1
        )

    def score(run: tuple[int, int]) -> tuple[int, int, int, float, int]:
        duration = interval(run)[2]
        ordered = any(c <= h for c in with_margin(run, contacts) for h in with_margin(run, holds))
        return (
            int(ordered),
            int(bool(inside(run, contacts)) and bool(inside(run, holds))),
            int(0.5 - 1e-12 <= duration <= 3.0 + 1e-12),
            duration,
            -run[0],
        )

    selected = max(hidden_runs, key=score) if hidden_runs else None
    selected_contact = selected_hold = None
    reappearance = None
    if selected is None:
        failures.append("full_occlusion_missing")
    else:
        duration = interval(selected)[2]
        if duration + 1e-12 < 0.5:
            failures.append("full_occlusion_duration_too_short")
        if duration > 3.0 + 1e-12:
            failures.append("full_occlusion_duration_too_long")
        for name, indices in (("fingerpad_contact", contacts), ("bilateral_hold", holds)):
            if not indices:
                failures.append(name + "_event_missing")
            elif not inside(selected, indices):
                failures.append(name + "_outside_selected_occlusion")
            elif not with_margin(selected, indices):
                failures.append(name + "_boundary_margin_failed")
        pairs = [
            (c, h)
            for c in with_margin(selected, contacts)
            for h in with_margin(selected, holds)
            if c <= h
        ]
        if pairs:
            selected_contact, selected_hold = pairs[0]
        elif with_margin(selected, contacts) and with_margin(selected, holds):
            failures.append("capture_event_order_invalid")
        contact_runs = {i for i, run in enumerate(hidden_runs) if inside(run, contacts)}
        hold_runs = {i for i, run in enumerate(hidden_runs) if inside(run, holds)}
        if contact_runs and hold_runs and contact_runs.isdisjoint(hold_runs):
            failures.append("capture_events_not_in_same_occlusion")
        post = selected[1] + 1
        post_runs = [
            (start + post, end + post)
            for start, end in runs([row[1] >= 0.8 for row in rows[post:]])
        ]
        adequate = [run for run in post_runs if run[1] - run[0] + 1 >= 3]
        reappearance = (
            adequate[0]
            if adequate
            else (max(post_runs, key=lambda x: x[1] - x[0]) if post_runs else None)
        )
        if not adequate:
            failures.append("reappearance_frames_insufficient")
    metrics = {
        "sample_count": len(rows),
        "initial_visible_frames": initial_frames,
        "initial_visible_duration_s": initial_duration,
        "initial_visible_displacement_m": initial_displacement,
        "hidden_interval_count": len(hidden_runs),
        "selected_hidden_frames": None if selected is None else interval(selected)[3],
        "selected_hidden_duration_s": None if selected is None else interval(selected)[2],
        "reappearance_frames": None if reappearance is None else interval(reappearance)[3],
    }
    return {
        "complete": not failures,
        "failure_codes": tuple(failures),
        "metrics": metrics,
        "initial_visible_interval": None if initial_end < 0 else (0, initial_end),
        "hidden_interval": selected,
        "reappearance_interval": reappearance,
        "fingerpad_contact_index": selected_contact,
        "bilateral_hold_index": selected_hold,
    }


class _OcclusionVisibilityAudit:
    """Evaluate the original visibility and capture-event protocol at 20 Hz."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rows: list[_OccRow] = []

    def observe(
        self,
        *,
        sensor_time_s: float,
        target_visible_ratio: float,
        target_position_world_m: tuple[float, float, float],
        fingerpad_contact_event: bool,
        bilateral_hold_event: bool,
    ) -> _OcclusionVisibilityState:
        timestamp_s = _finite(sensor_time_s, "sensor_time_s")
        frame_index = len(self._rows)
        expected_s = frame_index * _SENSOR_PERIOD_S
        if not math.isclose(timestamp_s, expected_s, rel_tol=0.0, abs_tol=_TIMESTAMP_TOLERANCE_S):
            raise ValueError("sensor_time_s must align with the next 20 Hz frame")
        ratio = _finite(target_visible_ratio, "target_visible_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("target_visible_ratio must be in [0, 1]")
        if len(target_position_world_m) != 3:
            raise ValueError("target_position_world_m must contain XYZ")
        position = cast(
            tuple[float, float, float],
            tuple(_finite(value, "target_position_world_m") for value in target_position_world_m),
        )
        self._rows.append(
            (
                timestamp_s,
                ratio,
                position,
                _strict_bool(fingerpad_contact_event, "fingerpad_contact_event"),
                _strict_bool(bilateral_hold_event, "bilateral_hold_event"),
            )
        )
        result = _occlusion_metrics(self._rows)
        metrics = cast(dict[str, object], result["metrics"])
        failures = cast(tuple[str, ...], result["failure_codes"])
        initial_frames = cast(int, metrics["initial_visible_frames"])
        hidden_frames = cast(int | None, metrics["selected_hidden_frames"]) or 0
        reappearance_frames = cast(int | None, metrics["reappearance_frames"]) or 0
        complete = cast(bool, result["complete"])
        initial_lost = initial_frames < len(self._rows) and any(
            name.startswith("initial_visible_") for name in failures
        )
        phase: _VisibilityPhase
        if complete:
            phase = "complete"
        elif initial_lost:
            phase = "failed"
        elif initial_frames == len(self._rows):
            phase = (
                "initial_visible" if initial_frames < _INITIAL_VISIBLE_FRAMES else "await_hidden"
            )
        elif ratio <= _FULLY_HIDDEN_RATIO:
            phase = "hidden"
        elif reappearance_frames:
            phase = "reappearance"
        elif hidden_frames:
            phase = "await_reappearance"
        else:
            phase = "await_hidden"
        return _OcclusionVisibilityState(
            frame_index=frame_index,
            sensor_time_s=timestamp_s,
            target_visible_ratio=ratio,
            phase=phase,
            initial_visible_frames=initial_frames,
            hidden_run_frames=hidden_frames,
            selected_hidden_duration_s=cast(float | None, metrics["selected_hidden_duration_s"]),
            reappearance_frames=reappearance_frames,
            complete=complete,
            failure_facts=_VisibilityFailureFacts(
                initial_visibility_lost=initial_lost,
                hidden_run_too_long="full_occlusion_duration_too_long" in failures,
            ),
            failure_codes=failures,
        )


@dataclass(frozen=True, slots=True)
class _CompiledOcclusionFacts:
    active_canopy_collision_enabled: bool
    lift_envelope_clearance_m: float
    robot_canopy_contact: bool
    target_canopy_contact: bool

    def __post_init__(self) -> None:
        _strict_bool(
            self.active_canopy_collision_enabled,
            "active_canopy_collision_enabled",
        )
        _finite(self.lift_envelope_clearance_m, "lift_envelope_clearance_m")
        _strict_bool(self.robot_canopy_contact, "robot_canopy_contact")
        _strict_bool(self.target_canopy_contact, "target_canopy_contact")


@dataclass(frozen=True, slots=True)
class _CompiledOcclusionFailureFacts:
    canopy_collision_disabled: bool
    lift_envelope_clearance_insufficient: bool
    unsafe_canopy_contact: bool


@dataclass(frozen=True, slots=True)
class _CompiledOcclusionAudit:
    safe: bool
    failure_facts: _CompiledOcclusionFailureFacts


def _validate_compiled_occlusion(
    facts: _CompiledOcclusionFacts,
) -> _CompiledOcclusionAudit:
    if not isinstance(facts, _CompiledOcclusionFacts):
        raise ValueError("facts must be _CompiledOcclusionFacts")
    failure_facts = _CompiledOcclusionFailureFacts(
        canopy_collision_disabled=not facts.active_canopy_collision_enabled,
        lift_envelope_clearance_insufficient=(
            facts.lift_envelope_clearance_m < _MINIMUM_LIFT_ENVELOPE_CLEARANCE_M
            and not math.isclose(
                facts.lift_envelope_clearance_m,
                _MINIMUM_LIFT_ENVELOPE_CLEARANCE_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        unsafe_canopy_contact=(facts.robot_canopy_contact or facts.target_canopy_contact),
    )
    safe = not any(
        (
            failure_facts.canopy_collision_disabled,
            failure_facts.lift_envelope_clearance_insufficient,
            failure_facts.unsafe_canopy_contact,
        )
    )
    return _CompiledOcclusionAudit(safe=safe, failure_facts=failure_facts)


def _canopy_model(path: Path, fixture: FixtureStateSpec) -> MujocoXMLObject:
    canopy = MujocoXMLObject(
        str(path),
        name=fixture.instance_id,
        joints=None,
        obj_type="all",
        duplicate_collision_geoms=False,
    )
    body = canopy.get_obj()
    body.set("pos", " ".join(format(value, ".17g") for value in fixture.position_world_m))
    x, y, z, w = fixture.quaternion_world_xyzw
    body.set("quat", " ".join(format(value, ".17g") for value in (w, x, y, z)))
    return canopy


def _read_canopy_fixture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    fixture: FixtureStateSpec,
    contact: ContactPhysicsSpec,
    body_id: int,
    geom_ids: tuple[int, ...],
) -> FixtureStateSpec:
    if data.model is not model:
        raise ValueError("canopy model/data ownership mismatch")
    if model.body_dofnum[body_id] != 0 or not geom_ids:
        raise ValueError("canopy must be a static collision fixture")
    expected = np.asarray(fixture.quaternion_world_xyzw, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    if not math.isclose(
        abs(float(np.dot(data.xquat[body_id][[1, 2, 3, 0]], expected))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("compiled canopy orientation differs from FixtureStateSpec")
    for geom_id in geom_ids:
        if model.geom_bodyid[geom_id] != body_id or np.any(model.geom_surfacevel[geom_id]):
            raise ValueError("canopy geoms must belong to its static non-driven body")
        actual = ContactPhysicsSpec(
            fixture.instance_id,
            cast(Vector3, tuple(float(v) for v in model.geom_friction[geom_id])),
            cast(Vector2, tuple(float(v) for v in model.geom_solref[geom_id])),
            cast(
                tuple[float, float, float, float, float],
                tuple(float(v) for v in model.geom_solimp[geom_id]),
            ),
        )
        if actual != contact:
            raise ValueError("compiled canopy contact properties differ from ScenarioSpec")
    return FixtureStateSpec(
        fixture.instance_id,
        fixture.asset_id,
        cast(Vector3, tuple(float(v) for v in data.xpos[body_id])),
        fixture.quaternion_world_xyzw,
    )


def _canopy_contact_facts(
    data: mujoco.MjData,
    *,
    canopy_geom_ids: frozenset[int],
    robot_geom_ids: frozenset[int],
    target_geom_ids: frozenset[int],
) -> tuple[bool, bool]:
    robot_contact = target_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & canopy_geom_ids:
            robot_contact = robot_contact or bool(pair & robot_geom_ids)
            target_contact = target_contact or bool(pair & target_geom_ids)
    return robot_contact, target_contact


def _compiled_lift_envelope_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_body_id: int,
    target_geom_ids: tuple[int, ...],
    canopy_geom_ids: tuple[int, ...],
    geometry: DrivenInterceptionGeometrySpec,
    minimum_lift_m: float,
) -> float:
    """Vertical space above the lifted target and below the canopy roof."""
    if data.model is not model or not target_geom_ids or not canopy_geom_ids:
        raise ValueError("lift envelope requires owned compiled target/canopy geometry")
    if _finite(minimum_lift_m, "minimum_lift_m") < 0.0:
        raise ValueError("minimum_lift_m must be non-negative")
    target_boxes = []
    for geom_id in target_geom_ids:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if (
            model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX
            or model.geom_bodyid[geom_id] != target_body_id
            or not np.allclose(np.abs(rotation).sum(axis=0), 1.0, rtol=0.0, atol=1e-12)
            or not np.allclose(np.abs(rotation).sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("T1-OCC envelope requires its initially axis-aligned collision boxes")
        extent = np.abs(rotation) @ model.geom_size[geom_id]
        target_boxes.append((data.geom_xpos[geom_id] - data.xipos[target_body_id], extent))
    bottom_offset = min(float(offset[2] - extent[2]) for offset, extent in target_boxes)
    top_offset = max(float(offset[2] + extent[2]) for offset, extent in target_boxes)
    lifted_top = geometry.support_surface_z_m + top_offset - bottom_offset + minimum_lift_m
    roofs = [
        geom_id
        for geom_id in canopy_geom_ids
        if model.geom(geom_id).name == "screen" or model.geom(geom_id).name.endswith("_screen")
    ]
    if len(roofs) != 1 or model.geom_type[roofs[0]] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("T1-OCC canopy must contain its declared screen roof")
    roof_id = roofs[0]
    roof_extent = np.abs(data.geom_xmat[roof_id].reshape(3, 3)) @ model.geom_size[roof_id]
    roof_bottom = float(data.geom_xpos[roof_id, 2] - roof_extent[2])
    return max(0.0, roof_bottom - lifted_top)


@dataclass(frozen=True, slots=True)
class _TargetVisibility:
    visible_pixels: int
    silhouette_pixels: int

    @property
    def ratio(self) -> float:
        if self.silhouette_pixels == 0:
            raise ValueError("agentview target silhouette is empty")
        return self.visible_pixels / self.silhouette_pixels


def _target_visibility(
    sim: Any, *, target_body_id: int, occluder_body_id: int
) -> _TargetVisibility:
    """Target coverage relative to the same scene with the canopy removed.

    Robosuite 1.5.2 returns [objtype, objid]. Keep robot and target occlusion
    in both rasters, as in the matched parked-canopy reference.
    An empty silhouette is an explicit raster fact, not a zero visibility ratio.
    """
    model = sim.model._model
    target_ids = np.flatnonzero(model.geom_bodyid == target_body_id)
    occluder_geom_ids = np.flatnonzero(model.geom_bodyid == occluder_body_id)
    context = sim._render_context_offscreen
    if context is None or not len(target_ids):
        raise ValueError("target visibility requires the live offscreen scene and target geoms")
    full = np.asarray(sim.render(camera_name="agentview", width=224, height=224, segmentation=True))
    visible = (full[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(full[:, :, 1], target_ids)
    selected = [
        copy.copy(geom)
        for geom in context.scn.geoms[: context.scn.ngeom]
        if not (geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in occluder_geom_ids)
    ]
    if not selected:
        raise ValueError("target has no rendered collision or visual geometry")
    for index, geom in enumerate(selected):
        destination = context.scn.geoms[index]
        mujoco.mjv_initGeom(
            destination,
            geom.type,
            np.asarray(geom.size, dtype=np.float64),
            np.asarray(geom.pos, dtype=np.float64),
            np.asarray(geom.mat, dtype=np.float64).reshape(9),
            geom.rgba,
        )
        destination.dataid = geom.dataid
        destination.objtype = geom.objtype
        destination.objid = geom.objid
        # Preserve each object's categorical ID in the canopy-free raster.
        destination.segid = geom.segid
    context.scn.ngeom = len(selected)
    context.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    context.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    try:
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 224, 224), context.scn, context.con)
        pixels = np.empty((224, 224, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, 224, 224), context.con)
    finally:
        context.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        context.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
    channels = pixels.astype(np.int32)
    encoded = channels[:, :, 0] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 2]
    silhouette = np.isin(
        encoded,
        [
            geom.segid + 1
            for geom in selected
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in target_ids
        ],
    )
    count = int(np.count_nonzero(silhouette))
    return _TargetVisibility(int(np.count_nonzero(visible & silhouette)), count)


__all__: list[str] = []
