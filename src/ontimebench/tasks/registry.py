from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.tokens import AliasToken, AnchorToken

from ontimebench._resources import resource_path
from ontimebench.tasks.spec import Family, TaskSpec, parse_task_spec


@dataclass(frozen=True, slots=True)
class TaskInfo:
    display_name: str
    paper_family: str
    implementation_family: Family


# Table S4 order also defines the order of scenes and scores in the test suite.
TASKS = {
    "harmonic_grasp": TaskInfo("Harmonic grasp", "Grasp", "grasp"),
    "conveyor_grasp": TaskInfo("Conveyor grasp", "Grasp", "driven_interception"),
    "translating_grasp": TaskInfo("Translating grasp", "Grasp", "driven_interception"),
    "rotating_grasp": TaskInfo("Rotating grasp", "Grasp", "driven_interception"),
    "occluded_grasp": TaskInfo("Occluded grasp", "Grasp", "driven_interception"),
    "moving_receptacle": TaskInfo("Moving receptacle", "Place", "moving_placement"),
    "rotating_receptacle": TaskInfo("Rotating receptacle", "Place", "moving_placement"),
    "wire_passage": TaskInfo("Wire passage", "Pass", "rotating_interaction"),
    "full_wire_passage": TaskInfo("Full wire passage", "Pass", "rotating_interaction"),
    "ball_capture": TaskInfo("Ball capture", "Capture", "free_dynamics_interception"),
    "ball_capture_on_table": TaskInfo(
        "Ball capture on table", "Capture", "free_dynamics_interception"
    ),
    "slider_capture": TaskInfo("Slider capture", "Capture", "free_dynamics_interception"),
    "can_capture": TaskInfo("Can capture", "Capture", "free_dynamics_interception"),
    "post_impact_grasp": TaskInfo("Post-impact grasp", "Impact", "replanning"),
}

TASK_IDS = tuple(TASKS)
TASK_FAMILIES: dict[str, Family] = {
    task_id: info.implementation_family for task_id, info in TASKS.items()
}


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictLoader, node: MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ConstructorError(
                None, None, "YAML merge keys are not allowed", key_node.start_mark
            )
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConstructorError(None, None, "mapping keys must be strings", key_node.start_mark)
        if key in mapping:
            raise ConstructorError(None, None, f"duplicate mapping key: {key}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml_strict(path: Path) -> Mapping[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        for token in yaml.scan(text):
            if isinstance(token, (AliasToken, AnchorToken)):
                raise ValueError(f"{path}: YAML anchors and aliases are not allowed")
        value = yaml.load(text, Loader=_StrictLoader)
    except (ValueError, yaml.YAMLError) as exc:
        if str(path) not in getattr(exc, "__notes__", ()):
            exc.add_note(str(path))
        raise
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: root must be a string-keyed mapping")
    return value


def load_task(task_id: str, config_dir: Path | None = None) -> TaskSpec:
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    try:
        family = TASKS[task_id].implementation_family
    except KeyError:
        raise ValueError(f"unknown task: {task_id}") from None
    directory = config_dir or resource_path("configs", "tasks")
    path = directory / f"{task_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return parse_task_spec(load_yaml_strict(path), task_id=task_id, family=family)
    except (ValueError, yaml.YAMLError) as exc:
        if str(path) not in getattr(exc, "__notes__", ()):
            exc.add_note(str(path))
        raise


def load_tasks(config_dir: Path | None = None) -> tuple[TaskSpec, ...]:
    return tuple(load_task(task_id, config_dir) for task_id in TASK_IDS)


__all__ = ["TASKS", "TASK_IDS", "load_task", "load_tasks"]
