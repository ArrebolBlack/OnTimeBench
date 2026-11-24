"""Run the held-out scene manifest and combine success rates across tasks."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path

from ontimebench._resources import resource_path
from ontimebench.evaluation.policy import ChunkPolicy, _chunk_policy
from ontimebench.evaluation.result import CONTROL_PERIOD_MS, EvaluationSummary, _summarize
from ontimebench.evaluation.rollout import TimingSink, _Replay, _rollout_episode
from ontimebench.scenarios.manifest import ScenarioManifest, load_manifest
from ontimebench.tasks.registry import TASK_IDS, load_task, load_yaml_strict


def _load_suite(path: str | Path | None) -> tuple[str, ScenarioManifest]:
    config = resource_path("configs", "suites", "test_id.yaml") if path is None else Path(path)
    spec = load_yaml_strict(config)
    if set(spec) != {"suite_id", "manifest"}:
        raise ValueError("a suite declares a suite_id and a manifest")
    suite_id, filename = spec["suite_id"], spec["manifest"]
    if (
        not isinstance(suite_id, str)
        or not suite_id
        or not isinstance(filename, str)
        or not filename
    ):
        raise ValueError("suite_id and manifest must be nonempty strings")
    manifest_path = (config.parent / filename).resolve()
    manifest = load_manifest(manifest_path)
    if not manifest.rows or any(row.partition != "test_id" for row in manifest):
        raise ValueError("evaluation requires held-out test_id scenes")
    if suite_id == "test_id":
        official = resource_path("manifests", "v1", "test_id.jsonl").resolve()
        if manifest_path != official and manifest_path.read_text(
            encoding="utf-8"
        ) != official.read_text(encoding="utf-8"):
            raise ValueError("use a distinct suite_id for a custom manifest")
        if Counter(row.task_id for row in manifest) != {task_id: 100 for task_id in TASK_IDS}:
            raise ValueError("test_id must contain 100 scenes per task")
    return suite_id, manifest


def _check_shard(index: int, count: int, scenes: int) -> None:
    if type(index) is not int or type(count) is not int or not 0 <= index < count <= scenes:
        raise ValueError("expected 0 <= shard_index < shard_count <= number of scenes")


def _published_latency_states(
    latency_seed: int, suite_id: str, manifest: ScenarioManifest
) -> dict[str, dict[str, int]]:
    """Recover the paper's per-scene streams when replaying its calibration seed."""
    if latency_seed != 20261023:
        return {}
    path = resource_path("manifests", "v1", "latency_streams.json")
    states: dict[str, dict[str, int]] = json.loads(path.read_text(encoding="utf-8"))
    # _load_suite already checked the identity of test_id. A custom suite may
    # reuse a published scene, or reuse its name for a different physical scene.
    reference = {}
    if suite_id != "test_id":
        official = load_manifest(resource_path("manifests", "v1", "test_id.jsonl"))
        reference = {row.scenario_id: row for row in official}
    return {
        row.scenario_id: states[row.scenario_id]
        for row in manifest
        if suite_id == "test_id" or reference.get(row.scenario_id) == row
    }


def evaluate(
    policy: ChunkPolicy,
    suite: str | Path | None = None,
    *,
    protocol: str,
    latency_profile: Mapping[str, object] | str | Path,
    target_pose_source: str,
    shard_index: int = 0,
    shard_count: int = 1,
    policy_settings: Mapping[str, object] | None = None,
    timing_sink: TimingSink | None = None,
    before_episode: Callable[[int], None] | None = None,
) -> EvaluationSummary:
    """Run one shard using the policy's measured batch-one query times.

    before_episode receives the scene's index in the full manifest, before the policy is reset.
    """
    profile = (
        json.loads(Path(latency_profile).read_text(encoding="utf-8"))
        if isinstance(latency_profile, (str, Path))
        else latency_profile
    )
    replay = _Replay.from_profile(protocol, profile)
    if target_pose_source not in ("none", "simulator"):
        raise ValueError("target_pose_source must be none or simulator")
    policy = _chunk_policy(policy)
    suite_id, manifest = _load_suite(suite)
    _check_shard(shard_index, shard_count, len(manifest))
    published_states = _published_latency_states(replay.seed, suite_id, manifest)
    action_spec = policy.action_spec
    settings = {
        "suite_id": suite_id,
        "protocol_version": manifest.rows[0].protocol_version,
        "name": replay.protocol,
        "target_pose_source": target_pose_source,
        "latency_samples_s": list(replay.samples_s),
        "latency_seed": replay.seed,
        "extra_latency_ms": replay.extra_ns / 1e6,
        "control_period_ms": CONTROL_PERIOD_MS,
        "action_spec": asdict(action_spec),
        "policy": dict(policy_settings)
        if policy_settings is not None
        else {
            "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        },
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    tasks = {}
    results = []
    for index in range(shard_index, len(manifest), shard_count):
        row = manifest.rows[index]
        if row.task_id not in tasks:
            tasks[row.task_id] = load_task(row.task_id)
        result = _rollout_episode(
            tasks[row.task_id],
            row.scenario,
            policy,
            scenario_id=row.scenario_id,
            partition=row.partition,
            factor=row.factor,
            repeat_index=0,
            source_seed=row.source_seed,
            execution_translation_limit_m=action_spec.motion_limits[0],
            target_pose_source=target_pose_source,
            replay=replay,
            initial_latency_state=published_states.get(row.scenario_id),
            timing_sink=timing_sink,
            before_reset=partial(before_episode, index) if before_episode is not None else None,
        )
        results.append(result)
    return _summarize(results, settings, len(manifest))
