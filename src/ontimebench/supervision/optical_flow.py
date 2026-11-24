"""Exact global dense 2D optical-flow supervision."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ontimebench.supervision.snapshot import SupervisionSnapshot

_DType = TypeVar("_DType", bound=np.generic)


def _readonly(value: npt.ArrayLike, dtype: np.dtype[_DType]) -> npt.NDArray[_DType]:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _exact_dense_optical_flow(
    source: SupervisionSnapshot,
    target: SupervisionSnapshot,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    if not isinstance(source, SupervisionSnapshot) or not isinstance(target, SupervisionSnapshot):
        raise ValueError("optical flow requires two SupervisionSnapshot records")
    if target.tick != source.tick + 1:
        raise ValueError("optical flow snapshots must have adjacent ticks")
    if target.timestamp_s <= source.timestamp_s:
        raise ValueError("optical flow target timestamp must follow the source")
    if source.view_names != target.view_names:
        raise ValueError("optical flow snapshots must have identical ordered views")
    if source.geom_semantic_identity != target.geom_semantic_identity:
        raise ValueError("optical flow snapshots must have identical geom identity")
    if source.world_from_geom.shape != target.world_from_geom.shape:
        raise ValueError("optical flow snapshots must have identical compiled geom count")

    views, height, width = source.depth_m.shape
    flow = np.zeros((views, height, width, 2), dtype=np.float32)
    valid = np.zeros((views, height, width), dtype=np.bool_)
    future_visible = np.zeros((views, height, width), dtype=np.bool_)
    pixel_u, pixel_v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    homogeneous_pixels = np.stack((pixel_u, pixel_v, np.ones_like(pixel_u)), axis=-1)
    flat_homogeneous_pixels = homogeneous_pixels.reshape(-1, 3)
    camera_unchanged = tuple(
        np.array_equal(source.camera_intrinsics[view], target.camera_intrinsics[view])
        and np.array_equal(source.world_from_camera_cv[view], target.world_from_camera_cv[view])
        for view in range(views)
    )
    geom_unchanged = np.all(source.world_from_geom == target.world_from_geom, axis=(1, 2))
    # Both views share compiled poses. Avoid a second grouping of the full pixel rasters.
    moving_geoms = (
        np.flatnonzero(~geom_unchanged) if all(camera_unchanged) else np.arange(len(geom_unchanged))
    )
    geom_from_world = {geom: np.linalg.inv(source.world_from_geom[geom]) for geom in moving_geoms}

    def _compute_view(view: int) -> None:
        source_geom_id = source.geom_id[view]
        source_depth = source.depth_m[view].astype(np.float64, copy=False)
        flat_source_depth = source_depth.reshape(-1)
        source_candidate = (source_geom_id >= 0) & (source_depth > 0.0)
        candidate_indices = np.flatnonzero(source_candidate)
        candidate_geoms = source_geom_id.reshape(-1)[candidate_indices]
        order = np.argsort(candidate_geoms, kind="stable")
        grouped_indices = candidate_indices[order]
        geom_ids, group_starts = np.unique(candidate_geoms[order], return_index=True)
        flat_flow = flow[view].reshape(-1, 2)
        flat_valid = valid[view].reshape(-1)
        flat_future_visible = future_visible[view].reshape(-1)
        moving_indices = grouped_indices[
            ~geom_unchanged[candidate_geoms[order]]
            if camera_unchanged[view]
            else np.ones(len(grouped_indices), dtype=np.bool_)
        ]
        if len(moving_indices):
            camera_points = (
                np.linalg.solve(
                    source.camera_intrinsics[view], flat_homogeneous_pixels[moving_indices].T
                ).T
                * flat_source_depth[moving_indices, None]
            )
            target_camera_from_world = np.linalg.inv(target.world_from_camera_cv[view])
        moving_offset = 0
        for group, geom_id in enumerate(geom_ids):
            geom = int(geom_id)
            group_end = (
                group_starts[group + 1] if group + 1 < len(geom_ids) else len(grouped_indices)
            )
            # Stable grouping retains the original row-major order within each geom.
            source_indices = grouped_indices[group_starts[group] : group_end]
            source_pixel_vectors = flat_homogeneous_pixels[source_indices]
            static_projection = camera_unchanged[view] and geom_unchanged[geom]
            source_selected = source_pixel_vectors[:, :2]
            if static_projection:
                target_pixels = source_selected
                projected_depth = flat_source_depth[source_indices]
            else:
                source_camera_points = camera_points[
                    moving_offset : moving_offset + len(source_indices)
                ]
                moving_offset += len(source_indices)
                source_world_points = (
                    source.world_from_camera_cv[view, :3, :3] @ source_camera_points.T
                ).T + source.world_from_camera_cv[view, :3, 3]
                source_geom_from_world = geom_from_world[geom]
                source_geom_points = (
                    source_geom_from_world[:3, :3] @ source_world_points.T
                ).T + source_geom_from_world[:3, 3]
                target_world_points = (
                    target.world_from_geom[geom, :3, :3] @ source_geom_points.T
                ).T + target.world_from_geom[geom, :3, 3]
                target_camera_points = (
                    target_camera_from_world[:3, :3] @ target_world_points.T
                ).T + target_camera_from_world[:3, 3]
                positive_depth = np.isfinite(target_camera_points).all(axis=1) & (
                    target_camera_points[:, 2] > 0.0
                )
                if not np.any(positive_depth):
                    continue
                source_indices = source_indices[positive_depth]
                source_selected = source_selected[positive_depth]
                projected_depth = target_camera_points[positive_depth, 2]
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    target_homogeneous = (
                        target.camera_intrinsics[view] @ target_camera_points[positive_depth].T
                    ).T
                    target_pixels = target_homogeneous[:, :2] / target_homogeneous[:, 2, None]

            finite_projection = np.isfinite(target_pixels).all(axis=1) & np.isfinite(
                projected_depth
            )
            if not np.any(finite_projection):
                continue
            if not np.all(finite_projection):
                source_indices = source_indices[finite_projection]
                source_selected = source_selected[finite_projection]
                target_pixels = target_pixels[finite_projection]
                projected_depth = projected_depth[finite_projection]
            with np.errstate(invalid="ignore", over="ignore"):
                projected_flow = np.asarray(target_pixels - source_selected, dtype=np.float32)
            representable = np.isfinite(projected_flow).all(axis=1)
            if not np.any(representable):
                continue
            if not np.all(representable):
                source_indices = source_indices[representable]
                source_selected = source_selected[representable]
                projected_depth = projected_depth[representable]
                projected_flow = projected_flow[representable]
            target_pixels_from_flow = source_selected + projected_flow.astype(
                np.float64, copy=False
            )
            flat_flow[source_indices] = projected_flow
            flat_valid[source_indices] = True

            in_frame = (
                (target_pixels_from_flow[:, 0] >= 0.0)
                & (target_pixels_from_flow[:, 0] <= width - 1)
                & (target_pixels_from_flow[:, 1] >= 0.0)
                & (target_pixels_from_flow[:, 1] <= height - 1)
            )
            if not np.any(in_frame):
                continue
            rounded = np.floor(target_pixels_from_flow[in_frame] + 0.5).astype(np.int64)
            target_u = rounded[:, 0]
            target_v = rounded[:, 1]
            visible_depth = projected_depth[in_frame]
            depth_tolerance = np.maximum(0.001, 0.001 * visible_depth)
            visible = (target.geom_id[view, target_v, target_u] == geom) & (
                np.abs(target.depth_m[view, target_v, target_u] - visible_depth) <= depth_tolerance
            )
            visible_sources = source_indices[in_frame][visible]
            flat_future_visible[visible_sources] = True

    if views == 1:
        _compute_view(0)
    else:
        error_settings = np.geterr()
        error_callback = np.geterrcall()

        def _compute_view_with_errors(view: int) -> None:
            previous_callback = np.geterrcall()
            try:
                with np.errstate(**error_settings):
                    np.seterrcall(error_callback)
                    _compute_view(view)
            finally:
                np.seterrcall(previous_callback)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_compute_view_with_errors, view) for view in range(views)]
            for future in futures:
                future.result()

    flow[~valid] = 0.0
    return (
        _readonly(flow, np.dtype(np.float32)),
        _readonly(valid, np.dtype(np.bool_)),
        _readonly(future_visible, np.dtype(np.bool_)),
    )


__all__: list[str] = []
