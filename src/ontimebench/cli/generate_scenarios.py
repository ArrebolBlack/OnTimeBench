from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ontimebench._resources import resource_path
from ontimebench.scenarios._rejection import ScenarioRejected
from ontimebench.scenarios.generator import generate_scenario
from ontimebench.scenarios.manifest import ScenarioManifest, ScenarioManifestRow, write_manifest
from ontimebench.tasks.registry import TASK_IDS, TASKS, load_task, load_yaml_strict


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate training scenarios for one task.")
    parser.add_argument("--task-id", required=True, choices=TASK_IDS)
    parser.add_argument("--seed", type=int, required=True, help="First proposal seed")
    parser.add_argument(
        "--num-seeds", type=int, required=True, help="Number of consecutive seeds to try"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Override the bundled training recipe")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.num_seeds < 1:
        parser.error("--seed must be non-negative and --num-seeds must be positive")
    output: Path = args.output
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if output.exists():
        raise FileExistsError(output)

    task = load_task(args.task_id)
    path = args.recipe or resource_path("configs", "scenarios", f"{task.task_id}.yaml")
    recipe = load_yaml_strict(path)
    rows: list[ScenarioManifestRow] = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        try:
            scene = generate_scenario(task, recipe, seed)
        except ScenarioRejected as exc:
            print(f"Skipping seed {seed}: {exc}", file=sys.stderr)
            continue
        rows.append(
            ScenarioManifestRow(
                schema_version=1,
                protocol_version="v1",
                scenario_id=f"v1:train:{task.task_id}:{len(rows):06d}",
                task_id=task.task_id,
                partition="train",
                factor=None,
                source_seed=seed,
                scenario=scene,
            )
        )
    if not rows:
        raise ValueError(f"All {args.num_seeds} proposed scenarios were rejected")
    write_manifest(ScenarioManifest(tuple(rows)), output)
    name = TASKS[task.task_id].display_name
    print(f"Wrote {len(rows)} of {args.num_seeds} proposed {name} scenarios to {output}")
