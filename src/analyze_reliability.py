#!/usr/bin/env python3
"""Analyze repeated-prompt reliability score CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_utils import REPO_ROOT, load_yaml, root_path, write_json_atomic


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def parse_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_group(group_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    responses = [normalized_text(row.get("response", "")) for row in rows]
    parsed_flags = [truthy(row.get("parse_success")) for row in rows]
    correct_flags = [truthy(row.get("correct")) for row in rows]
    parsed_predictions = [
        normalized_text(row.get("prediction", ""))
        for row in rows
        if truthy(row.get("parse_success"))
    ]
    numeric_predictions = [
        value
        for value in (parse_float(row.get("prediction")) for row in rows)
        if value is not None
    ]

    exact_response_agreement = len(set(responses)) <= 1
    parsed_answer_agreement = bool(parsed_predictions) and len(set(parsed_predictions)) == 1
    parse_consistent = len(set(parsed_flags)) <= 1
    accuracy_variance = float(np.var(np.asarray(correct_flags, dtype=float))) if rows else None
    numeric_std = float(np.std(numeric_predictions)) if numeric_predictions else None
    numeric_rel_range = None
    if numeric_predictions:
        denom = max(abs(float(np.mean(numeric_predictions))), 1e-12)
        numeric_rel_range = float((max(numeric_predictions) - min(numeric_predictions)) / denom)

    first = rows[0]
    return {
        "repeat_group_id": group_id,
        "model_label": first.get("model_label", ""),
        "suite": first.get("suite", ""),
        "answer_type": first.get("answer_type", ""),
        "encoding": first.get("encoding", ""),
        "feature": first.get("feature", ""),
        "statistic": first.get("statistic", ""),
        "repeats_observed": len(rows),
        "exact_response_agreement": exact_response_agreement,
        "parsed_answer_agreement": parsed_answer_agreement,
        "parse_consistent": parse_consistent,
        "parse_success_rate": float(np.mean(np.asarray(parsed_flags, dtype=float))) if rows else None,
        "accuracy_rate": float(np.mean(np.asarray(correct_flags, dtype=float))) if rows else None,
        "accuracy_variance": accuracy_variance,
        "numeric_prediction_std": numeric_std,
        "numeric_prediction_rel_range": numeric_rel_range,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_bool(field: str) -> float | None:
        values = [float(group[field]) for group in groups if isinstance(group.get(field), bool)]
        return float(np.mean(values)) if values else None

    def mean_float(field: str) -> float | None:
        values = [
            float(group[field])
            for group in groups
            if isinstance(group.get(field), (int, float)) and math.isfinite(float(group[field]))
        ]
        return float(np.mean(values)) if values else None

    return {
        "groups": len(groups),
        "mean_exact_response_agreement": mean_bool("exact_response_agreement"),
        "mean_parsed_answer_agreement": mean_bool("parsed_answer_agreement"),
        "mean_parse_consistency": mean_bool("parse_consistent"),
        "mean_parse_success_rate": mean_float("parse_success_rate"),
        "mean_accuracy_rate": mean_float("accuracy_rate"),
        "mean_accuracy_variance": mean_float("accuracy_variance"),
        "mean_numeric_prediction_std": mean_float("numeric_prediction_std"),
        "mean_numeric_prediction_rel_range": mean_float("numeric_prediction_rel_range"),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiments_config)
    evaluation_dir = REPO_ROOT / config["paths"]["evaluation_dir"]
    score_path = root_path(args.scores)
    rows = read_rows(score_path)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("repeat_group_id") or row.get("original_task_id")
        if group_id:
            grouped[group_id].append(row)
    if not grouped:
        raise SystemExit("No repeat_group_id values found in score CSV.")

    group_rows = [analyze_group(group_id, group_rows) for group_id, group_rows in sorted(grouped.items())]
    prefix = args.output_prefix or score_path.stem.replace("_scores", "")
    csv_path = evaluation_dir / f"{prefix}_reliability_groups.csv"
    summary_path = evaluation_dir / f"{prefix}_reliability_summary.json"
    write_csv(csv_path, group_rows)
    write_json_atomic(summary_path, summarize(group_rows))
    print(f"Wrote reliability groups: {csv_path}")
    print(f"Wrote reliability summary: {summary_path}")


if __name__ == "__main__":
    main()
