"""Evaluate a chunk policy using its measured batch-one query times."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from ontimebench.api import PROTOCOLS, check_output_path, evaluate, write_report
from ontimebench.evaluation.policy import _load_policy_factory
from ontimebench.evaluation.rollout import TimingSink


@contextmanager
def _timing_sink(path: Path | None) -> Iterator[TimingSink | None]:
    if path is None:
        yield None
        return
    stream = path.open("x", encoding="utf-8", newline="\n")
    try:
        with stream:

            def write_event(event: Mapping[str, object]) -> None:
                stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")

            yield write_event
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ontimebench-evaluate", description=__doc__)
    parser.add_argument(
        "--policy", required=True, help="importable module:factory returning a ChunkPolicy"
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument(
        "--latency-profile",
        required=True,
        type=Path,
        help="JSON file with measured batch-one latency_samples_s",
    )
    parser.add_argument(
        "--target-pose-source",
        choices=("none", "simulator"),
        required=True,
        help="whether to supply capture-aligned simulated target poses",
    )
    parser.add_argument("--suite", type=Path, help="defaults to the packaged test_id suite")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--policy-settings", type=Path, help="optional JSON identifying the model and settings"
    )
    parser.add_argument(
        "--timing-output", type=Path, help="optional per-query and applied-action JSONL trace"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    check_output_path(args.output)
    settings = {"factory": args.policy}
    if args.policy_settings is not None:
        parameters = json.loads(args.policy_settings.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict):
            raise ValueError("policy settings must be a JSON object")
        settings["parameters"] = parameters
    if args.timing_output is not None and args.output.resolve() == args.timing_output.resolve():
        parser.error("--output and --timing-output must be distinct")
    with _timing_sink(args.timing_output) as sink:
        summary = evaluate(
            _load_policy_factory(args.policy, chunked=True),
            args.suite,
            protocol=args.protocol,
            latency_profile=args.latency_profile,
            target_pose_source=args.target_pose_source,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            policy_settings=settings,
            timing_sink=sink,
        )
        write_report(summary, args.output)


if __name__ == "__main__":
    main()
