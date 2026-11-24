"""Labels for the single-camera E7 expert demonstrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ontimebench.control.action import ActionSpec
from ontimebench.envs.base import _E7OperationalSpaceController, _RecordingGeomFact


def _geometry(table: tuple[_RecordingGeomFact, ...], target_body: str) -> list[dict[str, object]]:
    rows = []
    for index, name, body, instance, asset in table:
        if body == target_body:
            roles = ["target_object"]
            if instance == "goal":
                body, roles = "goal_receptacle", ["goal_fixture", "target_object"]
            elif instance == "wire":
                body = "buzz_frame"
                wire = name.startswith(("wire_segment_", "wire_connector_"))
                roles = ["wire_fixture" if wire else "scene", "target_object"]
        elif instance in ("ramp", "gate"):
            roles = ["ramp_fixture"]
        elif instance in ("left_striker", "right_striker"):
            roles = ["striker"]
        elif instance == "occluder":
            roles = ["occluder"]
        else:
            roles = ["robot" if asset == "robot" else "scene"]
        rows.append(
            {"geom_id": index, "geom_name": name, "body_name": body, "semantic_identity": roles}
        )
    return rows


@dataclass
class TrainingEpisode:
    task_id: str
    seed: int
    world_deltas: list[npt.NDArray[np.float64]] = field(default_factory=list, init=False)
    valid_actions: int = field(default=0, init=False)

    @property
    def episode_id(self) -> str:
        return f"{self.task_id}-seed-{self.seed}"

    @property
    def path(self) -> Path:
        return Path("episodes", self.task_id, f"seed-{self.seed}.hdf5")

    def capture_goal(
        self, arm_input: npt.NDArray[np.float64], arm: _E7OperationalSpaceController
    ) -> None:
        # Sample before the first physics substep, while the achieved pose and goal share a tick.
        rotation = Rotation.from_matrix(arm.goal_ori @ arm.ref_ori_mat.T).as_rotvec()
        if arm_input[3:].any():
            nominal = np.clip(arm_input[3:], -1.0, 1.0) * 0.5
            if not np.allclose(rotation, nominal, atol=2e-7, rtol=0):
                raise ValueError("E7 rotation goal disagrees with the applied input")
            rotation = nominal
        self.world_deltas.append(np.r_[arm.goal_pos - arm.ref_pos, rotation])

    def prepare(
        self,
        arrays: dict[str, npt.NDArray[np.generic]],
        *,
        action_spec: ActionSpec,
        view_names: tuple[str, ...],
        frequency: int,
        event_times: Mapping[str, float],
        geoms: tuple[_RecordingGeomFact, ...],
        target_body: str,
    ) -> dict[str, object]:
        if (
            (action_spec.mode, action_spec.representation, action_spec.motion_limits)
            != ("E7", "delta", (0.05,) * 3 + (0.5,) * 3)
            or view_names != ("agentview",)
            or frequency != 20
        ):
            raise ValueError("training demonstrations require single-agentview E7 deltas at 20 Hz")
        issued = np.asarray(arrays["action_value"], dtype=np.float64)
        applied = np.asarray(arrays["applied_tick"], dtype=np.int64)
        valid = applied >= 0
        self.valid_actions = int(valid.sum())
        if len(self.world_deltas) != len(issued) or not valid[:-1].all() or not self.valid_actions:
            raise ValueError(
                "expected one controller goal per tick and at most one unapplied final action"
            )
        base = np.asarray(arrays["T_robot_base_from_world"], dtype=np.float64)[:3, :3]
        limits = np.asarray(action_spec.motion_limits)
        normalized = issued.copy()
        normalized[:, :3] = ((issued[:, :3] / limits[:3]) @ base).astype(np.float32)
        normalized[:, 3:6] = ((issued[:, 3:6] / limits[3:]) @ base).astype(np.float32)
        executed = np.stack(self.world_deltas)[applied[valid]]
        physical = np.zeros(issued.shape, dtype=np.float32)
        physical[valid, :3] = executed[:, :3] @ base.T
        physical[valid, 3:6] = executed[:, 3:] @ base.T
        physical[valid, 6] = issued[valid, 6]
        if (
            not np.isfinite(physical).all()
            or (np.abs(physical[:, :6]) > limits.astype(np.float32)).any()
        ):
            raise ValueError("applied E7 training label exceeds the physical action limits")
        arrays.update(action_value=physical, normalized_value=normalized, valid=valid)

        events = sorted(event_times)
        times = np.asarray(arrays["timestamp_s"], dtype=np.float64)
        flags = np.zeros((len(times), len(events)), dtype=bool)
        for column, name in enumerate(events):
            frame = np.searchsorted(times, event_times[name] - 1e-9)
            if frame == len(times):
                raise ValueError(f"event {name} occurred after the final observation")
            flags[frame, column] = True
        arrays["event_flags"] = flags
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "episode_seed": self.seed,
            "view_names": list(view_names),
            "control_frequency_hz": frequency,
            "action_spec": asdict(action_spec),
            "geom_table": _geometry(geoms, target_body),
            "event_names": events,
        }
