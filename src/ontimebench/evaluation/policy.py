from __future__ import annotations

from importlib import import_module
from typing import Literal, Protocol, cast, overload

from ontimebench.control.action import ActionSpec, BenchmarkAction
from ontimebench.sensing.observation import BenchmarkObservation


class BenchmarkPolicy(Protocol):
    """Single-command policy used when recording demonstrations."""

    def reset(self, task_id: str) -> None: ...

    def act(self, observation: BenchmarkObservation) -> BenchmarkAction: ...


class ChunkPolicy(Protocol):
    """Policy for measured-latency replay on the benchmark's control clock."""

    @property
    def action_spec(self) -> ActionSpec: ...

    def reset(self, task_id: str) -> None:
        """Start a scene of the given task, for example ``"harmonic_grasp"``."""
        ...

    def observe(self, observation: BenchmarkObservation) -> None:
        """Ingest a frame between queries."""
        ...

    def query_actions(self, observation: BenchmarkObservation) -> tuple[BenchmarkAction, ...]:
        """Ingest the query frame and return commands indexed from this frame's tick."""
        ...


def _step_policy(policy: object) -> BenchmarkPolicy:
    if not callable(getattr(policy, "reset", None)) or not callable(getattr(policy, "act", None)):
        raise ValueError("single-command policy requires callable reset and act")
    return cast(BenchmarkPolicy, policy)


def _chunk_policy(policy: object) -> ChunkPolicy:
    if not all(
        callable(getattr(policy, name, None)) for name in ("reset", "observe", "query_actions")
    ):
        raise ValueError("chunk policy requires callable reset, observe and query_actions")
    if not isinstance(getattr(policy, "action_spec", None), ActionSpec):
        raise ValueError("chunk policy must declare its Benchmark ActionSpec")
    return cast(ChunkPolicy, policy)


@overload
def _load_policy_factory(specification: str, *, chunked: Literal[True]) -> ChunkPolicy: ...


@overload
def _load_policy_factory(
    specification: str, *, chunked: Literal[False] = False
) -> BenchmarkPolicy: ...


def _load_policy_factory(
    specification: str, *, chunked: bool = False
) -> BenchmarkPolicy | ChunkPolicy:
    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError("policy must use the explicit module:factory form")
    module_name, factory_name = specification.split(":")
    if not module_name or not factory_name:
        raise ValueError("policy must use the explicit module:factory form")
    module = import_module(module_name)
    factory = cast(object, getattr(module, factory_name, None))
    if not callable(factory):
        raise ValueError("policy factory must be callable")
    policy = factory()
    return _chunk_policy(policy) if chunked else _step_policy(policy)


__all__ = ["BenchmarkPolicy", "ChunkPolicy"]
