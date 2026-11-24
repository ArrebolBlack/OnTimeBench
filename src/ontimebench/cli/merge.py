"""Combine all evaluation shards into one task-balanced report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ontimebench.api import check_output_path, write_report
from ontimebench.evaluation.result import (
    CONTROL_PERIOD_MS,
    EpisodeResult,
    EvaluationSummary,
    _summarize,
)
from ontimebench.evaluation.rollout import _Replay
from ontimebench.evaluation.suite import _check_shard, _load_suite


def _merge(suite_path: Path | None, shards: Sequence[Path]) -> EvaluationSummary:
    if not shards:
        raise ValueError("supply all evaluation shards")
    suite_id, manifest = _load_suite(suite_path)
    results: dict[str, EpisodeResult] = {}
    settings = None
    indices: set[int] = set()
    for path in shards:
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data["protocol"]
        index, count = protocol.pop("shard_index"), protocol.pop("shard_count")
        _check_shard(index, count, len(manifest))
        if count != len(shards) or index in indices:
            raise ValueError(f"{path}: supply one file for each of the {count} shards")
        indices.add(index)
        if (
            protocol["suite_id"] != suite_id
            or protocol["protocol_version"] != manifest.rows[0].protocol_version
        ):
            raise ValueError(f"{path}: shard does not match the requested suite")
        if settings is not None and settings != protocol:
            raise ValueError(f"{path}: policy, inputs, or timing differ between shards")
        if settings is None:
            settings = protocol
            replay = _Replay.from_profile(
                protocol["name"],
                {
                    "latency_samples_s": protocol["latency_samples_s"],
                    "seed": protocol["latency_seed"],
                },
            )
            if (protocol["extra_latency_ms"], protocol["control_period_ms"]) != (
                replay.extra_ns / 1e6,
                CONTROL_PERIOD_MS,
            ):
                raise ValueError(f"{path}: timing values do not match the selected protocol")
        expected = {row.scenario_id: row for row in manifest.rows[index::count]}
        for raw in data["results"]:
            raw["action_age_counts"] = tuple(raw["action_age_counts"])
            episode = EpisodeResult(**raw)
            row = expected.pop(episode.scenario_id, None)
            if row is None or (episode.task_id, episode.source_seed) != (
                row.task_id,
                row.source_seed,
            ):
                raise ValueError(f"{path}: unexpected or duplicate scene {episode.scenario_id}")
            if episode.target_pose_source != protocol["target_pose_source"]:
                raise ValueError(
                    f"{path}: scene target pose source differs from the declared input"
                )
            results[episode.scenario_id] = episode
        if expected:
            raise ValueError(f"{path}: missing {len(expected)} scenes from shard {index}")
    assert settings is not None
    ordered = tuple(results[row.scenario_id] for row in manifest)
    return _summarize(ordered, dict(settings, shard_index=0, shard_count=1), len(manifest))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ontimebench-merge", description=__doc__)
    parser.add_argument("--suite", type=Path, help="defaults to the packaged test_id suite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args(argv)
    check_output_path(args.output)
    write_report(_merge(args.suite, args.shards), args.output)


if __name__ == "__main__":
    main()
