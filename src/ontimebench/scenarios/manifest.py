from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ontimebench.scenarios.spec import (
    ScenarioSpec,
    family_for_geometry,
    scenario_from_dict,
    scenario_to_dict,
)
from ontimebench.tasks.registry import TASK_FAMILIES, TASK_IDS

Partition = Literal["train", "test_id"]


@dataclass(frozen=True, slots=True)
class ScenarioManifestRow:
    schema_version: int
    protocol_version: str
    scenario_id: str
    task_id: str
    partition: Partition
    factor: str | None
    source_seed: int
    scenario: ScenarioSpec

    def __post_init__(self) -> None:
        _validate_row_identity(
            self.schema_version,
            self.protocol_version,
            self.scenario_id,
            self.task_id,
            self.partition,
            self.factor,
            self.source_seed,
        )
        if not isinstance(self.scenario, ScenarioSpec):
            raise ValueError("scenario must be ScenarioSpec")
        if family_for_geometry(self.scenario.geometry) != TASK_FAMILIES[self.task_id]:
            raise ValueError("scenario geometry family does not match task_id")


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    rows: tuple[ScenarioManifestRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not all(
            isinstance(row, ScenarioManifestRow) for row in self.rows
        ):
            raise ValueError("manifest rows must be a tuple of ScenarioManifestRow")
        _validate_view_order(self.rows)
        if self.rows and any(row.partition != self.rows[0].partition for row in self.rows):
            raise ValueError("all manifest rows must belong to the same partition")

    def __iter__(self) -> Iterator[ScenarioManifestRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def shard(self, shard_index: int, shard_count: int) -> ScenarioManifest:
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count <= 0
            or shard_index < 0
            or shard_index >= shard_count
        ):
            raise ValueError(
                "invalid shard: require shard_count > 0 and 0 <= shard_index < shard_count"
            )
        return ScenarioManifest(
            tuple(row for index, row in enumerate(self.rows) if index % shard_count == shard_index)
        )


_ROW_FIELDS = {
    "schema_version",
    "protocol_version",
    "scenario_id",
    "task_id",
    "partition",
    "factor",
    "source_seed",
    "scenario",
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_row_identity(
    schema_version: object,
    protocol_version: object,
    scenario_id: object,
    task_id: object,
    partition: object,
    factor: object,
    source_seed: object,
) -> None:
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or not isinstance(protocol_version, str)
        or protocol_version != "v1"
    ):
        raise ValueError("manifest row requires schema_version 1 and protocol_version v1")
    if not isinstance(task_id, str) or task_id not in TASK_FAMILIES:
        raise ValueError(f"unknown task: {task_id}")
    if partition not in ("train", "test_id"):
        raise ValueError("invalid manifest partition")
    if factor is not None:
        raise ValueError("manifest factor must be null")
    if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
        raise ValueError("source_seed must be a non-negative integer")
    if not isinstance(scenario_id, str):
        raise ValueError("scenario_id must be a string")
    pattern = rf"v1:{partition}:{re.escape(task_id)}:\d{{6}}"
    if re.fullmatch(pattern, scenario_id) is None:
        raise ValueError(f"invalid scenario_id: {scenario_id}")


def _parse_row(value: object) -> ScenarioManifestRow:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("manifest row must be a string-keyed mapping")
    missing = _ROW_FIELDS - value.keys()
    unknown = value.keys() - _ROW_FIELDS
    if missing:
        raise ValueError(f"manifest row missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"manifest row unknown fields: {sorted(unknown)}")
    task_id = value["task_id"]
    partition = value["partition"]
    factor = value["factor"]
    source_seed = value["source_seed"]
    scenario_id = value["scenario_id"]
    _validate_row_identity(
        value["schema_version"],
        value["protocol_version"],
        scenario_id,
        task_id,
        partition,
        factor,
        source_seed,
    )
    scenario_value = value["scenario"]
    if not isinstance(scenario_value, Mapping):
        raise ValueError("scenario must be a mapping")
    return ScenarioManifestRow(
        schema_version=1,
        protocol_version="v1",
        scenario_id=scenario_id,
        task_id=task_id,
        partition=partition,
        factor=factor,
        source_seed=source_seed,
        scenario=scenario_from_dict(scenario_value, family=TASK_FAMILIES[task_id]),
    )


def _validate_view_order(rows: tuple[ScenarioManifestRow, ...]) -> None:
    task_order = {task_id: index for index, task_id in enumerate(TASK_IDS)}
    previous_task = -1
    previous_ordinals: dict[str, int] = {}
    identities: set[str] = set()
    for row in rows:
        if row.scenario_id in identities:
            raise ValueError(f"duplicate scenario_id: {row.scenario_id}")
        identities.add(row.scenario_id)
        current_task = task_order[row.task_id]
        if current_task < previous_task:
            raise ValueError("manifest rows must follow task registry order")
        previous_task = current_task
        actual = int(row.scenario_id.rsplit(":", 1)[1])
        previous = previous_ordinals.get(row.task_id)
        if previous is not None and actual <= previous:
            raise ValueError("manifest local ordinals must be strictly increasing within each task")
        previous_ordinals[row.task_id] = actual


def _validate_full_ordinals(rows: tuple[ScenarioManifestRow, ...]) -> None:
    ordinals: dict[str, int] = {}
    for row in rows:
        expected = ordinals.get(row.task_id, 0)
        actual = int(row.scenario_id.rsplit(":", 1)[1])
        if actual != expected:
            raise ValueError(f"manifest local ordinal for {row.task_id} must be {expected:06d}")
        ordinals[row.task_id] = expected + 1


def _validate_manifest_path_identity(manifest: ScenarioManifest, path: Path) -> None:
    if manifest.rows and path.stem == "test_id" and manifest.rows[0].partition != "test_id":
        raise ValueError("test_id.jsonl requires the test_id partition")


def load_manifest(path: str | Path) -> ScenarioManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[ScenarioManifestRow] = []
    text = source.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{source}:{line_number}: blank manifest rows are not allowed")
        try:
            value = json.loads(line, object_pairs_hook=_strict_object)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"{source}:{line_number}: {exc.msg}",
                exc.doc,
                exc.pos,
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
        try:
            rows.append(_parse_row(value))
        except (ValueError, NotImplementedError) as exc:
            exc.add_note(f"{source}:{line_number}")
            raise
    result = ScenarioManifest(tuple(rows))
    _validate_full_ordinals(result.rows)
    _validate_manifest_path_identity(result, source)
    return result


def write_manifest(manifest: ScenarioManifest, path: str | Path) -> None:
    if not isinstance(manifest, ScenarioManifest):
        raise ValueError("manifest must be ScenarioManifest")
    destination = Path(path)
    _validate_full_ordinals(manifest.rows)
    _validate_manifest_path_identity(manifest, destination)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for row in manifest.rows:
            payload = {
                "schema_version": row.schema_version,
                "protocol_version": row.protocol_version,
                "scenario_id": row.scenario_id,
                "task_id": row.task_id,
                "partition": row.partition,
                "factor": row.factor,
                "source_seed": row.source_seed,
                "scenario": scenario_to_dict(row.scenario),
            }
            stream.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")


__all__ = ["ScenarioManifest", "load_manifest", "write_manifest"]
