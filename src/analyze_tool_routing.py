#!/usr/bin/env python3
"""Create compact summaries for AI4Mat tool-routing score CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai4mat_common import REPO_ROOT, load_yaml


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
BOOL_COLUMNS = [
    "tool_json_valid",
    "tool_call_valid",
    "tool_executable",
    "tool_name_correct",
    "tool_image_args_correct",
    "tool_image_args_strict_correct",
    "tool_feature_correct",
    "tool_feature_strict_correct",
    "tool_statistic_correct",
    "tool_statistic_strict_correct",
    "tool_relation_correct",
    "tool_relation_strict_correct",
    "tool_top_k_correct",
    "tool_top_k_strict_correct",
    "tool_metric_scope_correct",
    "tool_metric_scope_strict_correct",
    "tool_routing_correct",
    "tool_routing_strict_correct",
    "tool_result_correct",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.lower()
    return lowered.map({"true": True, "1": True, "false": False, "0": False})


def mean_or_blank(df: pd.DataFrame, column: str) -> float | str:
    if column not in df.columns:
        return ""
    return df[column].mean()


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        answered = group[group["status"].astype(str).str.lower() == "success"]
        row = {col: value for col, value in zip(group_cols, group_key)}
        row["tasks"] = len(group)
        row["successful_outputs"] = len(answered)
        row["success_rate"] = len(answered) / len(group) if len(group) else 0.0
        row["valid_json_rate"] = mean_or_blank(answered, "tool_json_valid")
        row["valid_tool_call_rate"] = mean_or_blank(answered, "tool_call_valid")
        row["executable_rate"] = mean_or_blank(answered, "tool_executable")
        row["routing_accuracy"] = mean_or_blank(answered, "tool_routing_correct")
        row["strict_routing_accuracy"] = mean_or_blank(answered, "tool_routing_strict_correct")
        row["result_accuracy"] = mean_or_blank(answered, "tool_result_correct")
        row["end_to_end_accuracy"] = mean_or_blank(answered, "correct")
        row["coverage_correct"] = mean_or_blank(group, "correct")
        row["mean_latency_s"] = pd.to_numeric(answered["elapsed_seconds"], errors="coerce").mean()
        row["mean_total_tokens"] = pd.to_numeric(answered["total_tokens"], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def argument_failures(df: pd.DataFrame) -> pd.DataFrame:
    argument_cols = [
        "tool_name_correct",
        "tool_image_args_correct",
        "tool_image_args_strict_correct",
        "tool_feature_correct",
        "tool_feature_strict_correct",
        "tool_statistic_correct",
        "tool_statistic_strict_correct",
        "tool_relation_correct",
        "tool_relation_strict_correct",
        "tool_top_k_correct",
        "tool_top_k_strict_correct",
        "tool_metric_scope_correct",
        "tool_metric_scope_strict_correct",
    ]
    rows = []
    for (model, task_type), group in df.groupby(["model_label", "tool_task_type"], dropna=False):
        answered = group[group["status"].astype(str).str.lower() == "success"]
        row = {
            "model_label": model,
            "tool_task_type": task_type,
            "tasks": len(group),
            "successful_outputs": len(answered),
        }
        for col in argument_cols:
            if col not in answered.columns:
                row[f"{col}_failure_rate"] = ""
                continue
            values = answered[col]
            values = values[values.notna()]
            if values.empty:
                row[f"{col}_failure_rate"] = ""
            else:
                row[f"{col}_failure_rate"] = 1.0 - values.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiments_config)
    evaluation_dir = REPO_ROOT / config["paths"]["evaluation_dir"]
    scores_path = REPO_ROOT / args.scores
    prefix = args.output_prefix or scores_path.stem.replace("_scores", "")

    df = pd.read_csv(scores_path)
    df = df[df["answer_type"] == "tool_call"].copy()
    if df.empty:
        raise SystemExit("No tool_call rows found in score CSV.")
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = bool_series(df[col])

    outputs = {
        "tool_routing_by_model": summarize(df, ["model_label"]),
        "tool_routing_by_task_type": summarize(df, ["model_label", "tool_task_type"]),
        "tool_routing_by_paraphrase": summarize(
            df,
            ["model_label", "tool_task_type", "paraphrase_family"],
        ),
        "tool_routing_argument_failures": argument_failures(df),
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        path = evaluation_dir / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
