#!/usr/bin/env python3
"""Build repeated-prompt reliability manifests for AI4Mat runs."""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from typing import Any

from ai4mat_common import iter_jsonl, load_yaml, write_jsonl


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def stratum_key(task: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(task.get("suite", "unknown")),
        str(task.get("encoding", "unknown")),
        str(task.get("answer_type", "unknown")),
    )


def stratified_sample(
    tasks: list[dict[str, Any]],
    *,
    subset_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size >= len(tasks):
        return list(tasks)

    indexed_groups: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        indexed_groups[stratum_key(task)].append((index, task))

    rng = random.Random(seed)
    total = len(tasks)
    allocations: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key, group in indexed_groups.items():
        exact = subset_size * (len(group) / total)
        base = min(len(group), int(math.floor(exact)))
        allocations[key] = base
        remainders.append((exact - base, key))

    remainders.sort(reverse=True)
    while sum(allocations.values()) < subset_size:
        progressed = False
        for _, key in remainders:
            if allocations[key] < len(indexed_groups[key]):
                allocations[key] += 1
                progressed = True
                if sum(allocations.values()) >= subset_size:
                    break
        if not progressed:
            break

    selected: list[tuple[int, dict[str, Any]]] = []
    for key, group in indexed_groups.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: allocations[key]])

    selected.sort(key=lambda item: item[0])
    return [task for _, task in selected[:subset_size]]


def repeated_records(tasks: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        original_task_id = str(task["task_id"])
        for repeat_index in range(1, repeat_count + 1):
            repeated = dict(task)
            repeated["task_id"] = f"{original_task_id}__repeat{repeat_index:02d}"
            repeated["repeat_group_id"] = original_task_id
            repeated["original_task_id"] = original_task_id
            repeated["repeat_index"] = repeat_index
            repeated["repeat_count"] = repeat_count
            records.append(repeated)
    return records


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiments_config)
    reliability = config.get("reliability", {})
    subset_size = int(args.subset_size or reliability.get("stratified_subset_size", 256))
    repeat_count = int(args.repeat_count or reliability.get("repeat_count", 3))
    seed = int(args.seed or reliability.get("selection_seed", config["project"]["random_seed"]))

    tasks = list(iter_jsonl(args.source_manifest))
    selected = stratified_sample(tasks, subset_size=subset_size, seed=seed)
    records = repeated_records(selected, repeat_count)
    count = write_jsonl(args.output, records)
    print(
        f"Wrote {count:,} repeated tasks from {len(selected):,} source tasks "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
