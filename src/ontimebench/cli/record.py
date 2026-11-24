from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from ontimebench.evaluation.policy import _load_policy_factory
from ontimebench.evaluation.rollout import _rollout_episode
from ontimebench.scenarios.manifest import ScenarioManifest, load_manifest
from ontimebench.tasks.registry import load_task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ontimebench-record")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--policy", required=True, help="module:factory or the built-in offline native-oracle"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--execution-translation-limit-m", type=float, choices=(0.05, 0.10), default=0.05
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--training",
        action="store_true",
        help="write successful native-expert episodes and train.jsonl to a new directory",
    )
    return parser


def _record_training(scenes: ScenarioManifest, output: Path) -> None:
    from ontimebench.data.training import TrainingEpisode

    if not scenes or any(
        row.partition != "train"
        or row.scenario.action_mode != "E7"
        or row.scenario.observation_profile != "agentview"
        or row.scenario.control_frequency_hz != 20
        or row.scenario.action_delay_steps > 1
        for row in scenes
    ):
        raise ValueError(
            "expected train scenes with agentview, 20 Hz E7 actions and at most one tick of delay"
        )
    if output.exists():
        raise FileExistsError(f"choose a new training output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".ontimebench-") as temporary:
        dataset = Path(temporary) / "dataset"
        dataset.mkdir()
        with (dataset / "train.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in scenes:
                episode = TrainingEpisode(row.task_id, row.source_seed)
                result = _rollout_episode(
                    load_task(row.task_id),
                    row.scenario,
                    None,
                    scenario_id=row.scenario_id,
                    partition=row.partition,
                    factor=row.factor,
                    repeat_index=0,
                    source_seed=row.source_seed,
                    record_directory=dataset,
                    training_episode=episode,
                    _native_oracle=True,
                )
                record = {
                    "episode_id": episode.episode_id,
                    "task_id": row.task_id,
                    "path": episode.path.as_posix(),
                    "frames": result.episode_steps + 1,
                    "valid_actions": episode.valid_actions,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        dataset.rename(output)
    print(f"Wrote {len(scenes)} demonstrations to {output / 'train.jsonl'}")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    output: Path = arguments.output
    manifest = load_manifest(arguments.manifest)
    shard = manifest.shard(arguments.shard_index, arguments.shard_count)
    native_oracle = arguments.policy == "native-oracle"
    if arguments.training:
        if not native_oracle or arguments.execution_translation_limit_m != 0.05:
            raise ValueError("--training requires native-oracle with the 0.05 m training limit")
        if arguments.shard_count > 1:
            output = output / f"shard-{arguments.shard_index:05d}-of-{arguments.shard_count:05d}"
        _record_training(shard, output)
        return
    if not output.is_dir():
        raise ValueError("recording output must be an existing directory")
    policy = None if native_oracle else _load_policy_factory(arguments.policy)
    for row in shard:
        _rollout_episode(
            load_task(row.task_id),
            row.scenario,
            policy,
            scenario_id=row.scenario_id,
            partition=row.partition,
            factor=row.factor,
            repeat_index=0,
            source_seed=row.source_seed,
            record_directory=output,
            execution_translation_limit_m=arguments.execution_translation_limit_m,
            _native_oracle=native_oracle,
        )


__all__ = ["main"]
