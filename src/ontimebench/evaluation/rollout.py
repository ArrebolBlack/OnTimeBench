"""Apply observation-indexed action chunks at 20 Hz on the simulation clock."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ontimebench.control.action import BenchmarkAction, validate_action_spec
from ontimebench.evaluation.policy import BenchmarkPolicy, ChunkPolicy, _chunk_policy, _step_policy
from ontimebench.evaluation.result import CONTROL_PERIOD_MS, EpisodeResult
from ontimebench.scenarios.manifest import Partition
from ontimebench.scenarios.spec import ScenarioSpec
from ontimebench.sensing.observation import BenchmarkObservation
from ontimebench.tasks.spec import TaskSpec

if TYPE_CHECKING:
    from ontimebench.data.training import TrainingEpisode

_LOGGER = logging.getLogger(__name__)
_PERIOD_NS = CONTROL_PERIOD_MS * 1_000_000
PROTOCOLS = ("native", "plus150ms", "paused")
TimingSink = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class _Replay:
    protocol: str
    samples_s: tuple[float, ...]
    seed: int

    @classmethod
    def from_profile(cls, protocol: str, values: Mapping[str, object]) -> _Replay:
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        if not isinstance(values, Mapping) or set(values) - {"latency_samples_s", "seed"}:
            raise ValueError("profile must contain latency_samples_s and an optional seed")
        samples = values.get("latency_samples_s")
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError("supply a nonempty batch-one latency_samples_s array")
        if any(
            isinstance(s, bool) or not isinstance(s, (float, int)) or not math.isfinite(s) or s < 0
            for s in samples
        ):
            raise ValueError("latency samples must be finite, non-negative seconds")
        seed = values.get("seed", 0)
        if type(seed) is not int or seed < 0:
            raise ValueError("latency seed must be a non-negative integer")
        return cls(protocol, tuple(float(s) for s in samples), seed)

    @property
    def extra_ns(self) -> int:
        return 150_000_000 if self.protocol == "plus150ms" else 0


class _LatencyExecutor:
    def __init__(
        self,
        policy: ChunkPolicy,
        replay: _Replay,
        scenario_id: str,
        repeat_index: int,
        emit: TimingSink | None = None,
        initial_latency_state: dict[str, int] | None = None,
    ) -> None:
        self.policy = policy
        self.replay = replay
        self.emit = emit
        if initial_latency_state is not None and repeat_index == 0:
            # The published PCG64 streams start before the first 32-bit draw.
            bits = np.random.PCG64(0)
            bits.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": initial_latency_state["state"],
                    "inc": initial_latency_state["inc"],
                },
                "has_uint32": 0,
                "uinteger": 0,
            }
        else:
            seed = np.random.SeedSequence([replay.seed, repeat_index, *scenario_id.encode("utf-8")])
            bits = np.random.PCG64(seed)
        self.rng = np.random.Generator(bits)
        self.pending: tuple[int, int, tuple[BenchmarkAction, ...]] | None = None
        self.active: tuple[BenchmarkAction, ...] = ()
        self.capture_tick = 0
        self.queued_captures: dict[int, int] = {}
        self.ages: Counter[int] = Counter()

    def _adopt(self, capture: int, chunk: tuple[BenchmarkAction, ...], tick: int) -> None:
        self.active, self.capture_tick = chunk, capture
        if self.emit is not None:
            slot = tick - capture
            self.emit(
                {
                    "event": "benchmark_chunk_adopted",
                    "tick": tick,
                    "observation_tick": capture,
                    "first_valid_command_index": slot if slot < len(chunk) else None,
                }
            )

    def action(self, observation: BenchmarkObservation) -> BenchmarkAction | None:
        tick = observation.tick
        now_ns = tick * _PERIOD_NS
        if not math.isclose(observation.timestamp_s, now_ns / 1e9, abs_tol=1e-10, rel_tol=0):
            raise ValueError("latency replay requires aligned 20 Hz observations")
        if self.pending is not None and self.pending[0] <= now_ns:
            _, capture, chunk = self.pending
            self._adopt(capture, chunk, tick)
            self.pending = None
        if self.pending is None:
            index = int(self.rng.integers(len(self.replay.samples_s)))
            intrinsic_s = self.replay.samples_s[index]
            delay_ns = round(intrinsic_s * 1e9) + self.replay.extra_ns
            chunk = self.policy.query_actions(observation)
            if not isinstance(chunk, tuple) or not chunk:
                raise ValueError("query_actions must return a nonempty tuple of BenchmarkAction")
            for action in chunk:
                validate_action_spec(action, self.policy.action_spec)
            if self.emit is not None:
                self.emit(
                    {
                        "event": "benchmark_query",
                        "observation_tick": tick,
                        "sample_index": index,
                        "inference_latency_s": intrinsic_s,
                        "extra_latency_s": self.replay.extra_ns / 1e9,
                        "chunk_length": len(chunk),
                    }
                )
            if self.replay.protocol == "paused":
                self._adopt(tick, chunk, tick)
            else:
                self.pending = now_ns + max(_PERIOD_NS, delay_ns), tick, chunk
        else:
            self.policy.observe(observation)

        # Slot j predicts the action at capture_tick + j. Expired slots are skipped.
        slot = tick - self.capture_tick
        if slot >= len(self.active):
            return None
        self.queued_captures[tick] = self.capture_tick
        return self.active[slot]

    def record_application(self, tick: int, fact: tuple[int, int, BenchmarkAction] | None) -> None:
        if fact is None:
            return
        issued, applied, _ = fact
        if applied != tick:
            raise RuntimeError("applied action did not match the simulation tick")
        capture = self.queued_captures.pop(issued)
        self.ages[applied - capture] += 1
        if self.emit is not None:
            self.emit(
                {
                    "event": "benchmark_action_timing",
                    "observation_tick": capture,
                    "issued_tick": issued,
                    "applied_tick": applied,
                    "action_age_ms": (applied - capture) * CONTROL_PERIOD_MS,
                }
            )


def _read_only(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _rollout_episode(
    task: TaskSpec,
    scenario: ScenarioSpec,
    policy: BenchmarkPolicy | ChunkPolicy | None,
    *,
    scenario_id: str,
    partition: Partition,
    factor: str | None,
    repeat_index: int,
    source_seed: int = 0,
    record_directory: str | Path | None = None,
    training_episode: TrainingEpisode | None = None,
    execution_translation_limit_m: float = 0.05,
    _native_oracle: bool = False,
    replay: _Replay | None = None,
    initial_latency_state: dict[str, int] | None = None,
    target_pose_source: str = "none",
    timing_sink: TimingSink | None = None,
    before_reset: Callable[[], None] | None = None,
) -> EpisodeResult:
    if (replay is None) != (record_directory is not None):
        raise ValueError("scoring requires a timing profile; recording uses synchronous actions")
    if _native_oracle and (
        policy is not None or replay is not None or execution_translation_limit_m != 0.05
    ):
        raise ValueError("native-oracle recording requires its 0.05 m action profile")
    if target_pose_source not in ("none", "simulator"):
        raise ValueError("target_pose_source must be none or simulator")

    from ontimebench.envs.factory import make_env

    env = make_env(
        task.task_id, scenario, execution_translation_limit_m=execution_translation_limit_m
    )
    try:

        def observe(raw_observation: Mapping[str, object]) -> BenchmarkObservation:
            observation = env._benchmark_observation(raw_observation)
            if target_pose_source == "none":
                return observation
            # Read at the capture tick, before the policy sees this frame.
            position, quaternion = env._target_pose_in_robot_base(observation)
            return replace(
                observation,
                target_position=_read_only(position),
                target_quaternion_xyzw=_read_only(quaternion),
            )

        recorder = None
        if record_directory is not None:
            from ontimebench.data.recorder import EpisodeRecorder

            recorder = EpisodeRecorder(
                env,
                record_directory,
                task=task,
                scenario=scenario,
                scenario_id=scenario_id,
                partition=partition,
                factor=factor,
                repeat_index=repeat_index,
                training=training_episode,
            )
        raw = env.reset() if recorder is None else recorder.reset()
        executor = None
        oracle = None
        step_policy = None
        if replay is not None:
            chunk_policy = _chunk_policy(policy)
            emit = None
            log_timing = _LOGGER.isEnabledFor(logging.INFO)
            if timing_sink is not None or log_timing:

                def emit(row: Mapping[str, object]) -> None:
                    event = {"scenario_id": scenario_id, "protocol": replay.protocol, **row}
                    if timing_sink is not None:
                        timing_sink(event)
                    if log_timing:
                        _LOGGER.info("%s", json.dumps(event, separators=(",", ":")))

            executor = _LatencyExecutor(
                chunk_policy, replay, scenario_id, repeat_index, emit, initial_latency_state
            )
            env._enable_asynchronous_actions(chunk_policy.action_spec)
        elif _native_oracle:
            from ontimebench.scenarios._native_oracle import _make_native_oracle

            oracle = _make_native_oracle(env)
        else:
            step_policy = _step_policy(policy)

        if policy is not None:
            if before_reset is not None:
                before_reset()
            policy.reset(task.task_id)
        observation = observe(raw)
        if recorder is not None:
            recorder.capture_reset_frame(observation)
        info: Mapping[str, object]
        for tick in range(task.episode_horizon_steps):
            if observation.tick != tick:
                raise RuntimeError("observation tick does not match the rollout tick")
            if executor is not None:
                action = executor.action(observation)
            elif oracle is not None:
                action = oracle.action()
            else:
                assert step_policy is not None
                action = step_policy.act(observation)
            if recorder is None:
                raw, _, done, info = env.step(action)
            else:
                if action is None:
                    raise ValueError("recording requires one action per control tick")
                raw, _, done, info = recorder.step(action)
            success = info["success"]
            if not isinstance(success, bool):
                raise RuntimeError("episode success must be a boolean")
            if executor is not None:
                executor.record_application(tick, env._last_action_application_fact())
            if oracle is not None:
                oracle.observe_step(done=done, success=success)
            if not done or recorder is not None:
                observation = observe(raw)
                if recorder is not None:
                    recorder.capture_post_step_frame(observation)
            if done:
                break
        else:
            raise RuntimeError("episode did not terminate within the task horizon")

        stages = info["stage_facts"]
        elapsed, failure = info["terminal_time_s"], info["failure_code"]
        if not isinstance(stages, tuple) or not stages or not isinstance(elapsed, (int, float)):
            raise RuntimeError("terminal episode is missing stages or elapsed simulation time")
        if failure is not None and not isinstance(failure, str):
            raise RuntimeError("invalid terminal failure code")
        completed = 0.0
        for _, reached, progress, _ in stages:
            if not reached:
                completed += progress
                break
            completed += 1
        if recorder is not None:
            if env._action_spec is None:
                raise RuntimeError("recorded episode did not latch an action specification")
            recorder.finalize_episode(
                action_spec=env._action_spec,
                event_names=tuple(stage[0] for stage in stages),
                success=success,
                failure_code=failure,
                terminal_time_s=elapsed,
            )
        ages = (
            ()
            if executor is None
            else tuple(executor.ages[i] for i in range(max(executor.ages, default=-1) + 1))
        )
        return EpisodeResult(
            scenario_id,
            task.task_id,
            source_seed,
            target_pose_source,
            success,
            failure,
            tick + 1,
            elapsed,
            completed / len(stages),
            ages,
        )
    finally:
        env.close()
