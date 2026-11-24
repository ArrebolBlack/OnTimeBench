"""Structured rejection facts shared by reset and native oracle checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ontimebench.envs.components.occlusion import _OcclusionVisibilityState


class ScenarioRejected(ValueError):
    def __init__(
        self,
        stage: Literal["reset", "visibility", "oracle"],
        failure_code: str,
        *,
        task_id: str,
        terminal_time_s: float | None = None,
        events: Mapping[str, float] | None = None,
        visibility_state: _OcclusionVisibilityState | None = None,
    ) -> None:
        self.stage = stage
        self.failure_code = failure_code
        self.terminal_time_s = terminal_time_s
        self.events = dict(events) if events is not None else {}
        self.visibility_state = visibility_state
        super().__init__(f"{task_id} rejected at {stage}: {failure_code}")
