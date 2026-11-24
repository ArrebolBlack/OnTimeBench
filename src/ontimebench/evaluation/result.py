"""Episode outcomes and the task-balanced score of a completed suite."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ontimebench.tasks.registry import TASKS

FAILURE_CODES = (
    "invalid_state",
    "unsafe_contact",
    "workspace_exit",
    "downstream_exit",
    "drop",
    "driver_blocked",
    "timeout",
    "forced_insertion",
    "goal_window_expired",
    "wire_contact",
    "stage_block",
    "path_deviation",
    "path_backtrack",
    "premature_release",
    "release_impulse",
    "early_robot_contact",
    "tray_contact",
    "stop_or_deflect_only",
    "unstable_tipping",
    "gripper_support_contact",
    "no_selected_impact",
    "extra_impact",
    "wrong_striker_contact",
    "insufficient_outgoing_change",
    "striker_trap",
)

CONTROL_PERIOD_MS = 50


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    task_id: str
    source_seed: int
    target_pose_source: str
    success: bool
    failure_code: str | None
    episode_steps: int
    elapsed_s: float
    task_completion: float
    # Entry i counts commands applied i control ticks after their observation.
    action_age_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.target_pose_source not in ("none", "simulator"):
            raise ValueError("target_pose_source must be none or simulator")
        if type(self.success) is not bool or self.success != (self.failure_code is None):
            raise ValueError("a failed episode requires a failure code")
        if self.failure_code is not None and self.failure_code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code: {self.failure_code}")
        if type(self.episode_steps) is not int or self.episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and non-negative")
        if not 0 <= self.task_completion <= 1 or self.success != (self.task_completion == 1):
            raise ValueError("task completion and success are inconsistent")
        if type(self.source_seed) is not int or self.source_seed < 0:
            raise ValueError("source_seed must be a non-negative integer")
        if (
            len(self.action_age_counts) > self.episode_steps
            or any(type(n) is not int or n < 0 for n in self.action_age_counts)
            or sum(self.action_age_counts) > self.episode_steps
        ):
            raise ValueError("invalid counts of applied actions by age")


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    display_name: str
    task_family: str
    successes: int
    scenes: int
    success_rate: float


@dataclass(frozen=True)
class EvaluationSummary:
    protocol: dict[str, object]
    evaluated_scenes: int
    suite_scenes: int
    per_task: tuple[TaskResult, ...]
    task_balanced_success_rate: float | None
    query_latency_p95_ms: float
    action_age_p95_ms: float | None
    results: tuple[EpisodeResult, ...]


def check_output_path(path: Path) -> None:
    """Check that a new report can be written in an existing directory."""
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if path.exists():
        raise FileExistsError(path)


def write_report(summary: EvaluationSummary, output: Path) -> None:
    """Write the suite report to a new JSON file and print the success count."""
    payload = json.dumps(asdict(summary), allow_nan=False, separators=(",", ":"))
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
    for task in summary.per_task:
        print(f"{task.display_name}: {task.successes}/{task.scenes}")
    successes = sum(row.success for row in summary.results)
    print(f"{successes}/{summary.evaluated_scenes} successful scenes; report saved to {output}")


def _summarize(
    results: Sequence[EpisodeResult], protocol: dict[str, object], suite_scenes: int
) -> EvaluationSummary:
    by_task: dict[str, list[bool]] = defaultdict(list)
    ages: Counter[int] = Counter()
    for row in results:
        by_task[row.task_id].append(row.success)
        ages.update(dict(enumerate(row.action_age_counts)))

    per_task = tuple(
        TaskResult(
            task_id,
            info.display_name,
            info.paper_family,
            sum(values),
            len(values),
            sum(values) / len(values),
        )
        for task_id, info in TASKS.items()
        if (values := by_task.get(task_id))
    )
    score = None
    if len(results) == suite_scenes:
        score = sum(row.success_rate for row in per_task) / len(per_task)
    action_age = None
    if ages.total():
        samples = np.repeat(list(ages), list(ages.values()))
        action_age = float(np.percentile(samples, 95)) * CONTROL_PERIOD_MS
    calibration = np.asarray(protocol["latency_samples_s"], dtype=np.float64)
    return EvaluationSummary(
        protocol,
        len(results),
        suite_scenes,
        per_task,
        score,
        float(np.percentile(calibration, 95)) * 1000,
        action_age,
        tuple(results),
    )
