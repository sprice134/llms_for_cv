#!/usr/bin/env python3
"""Build reviewer-facing candidate analyses without changing main paper artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ai4mat_common import REPO_ROOT, iter_jsonl, load_yaml, slugify
from evaluate_numeric import score_tool_call


OUTPUT_DIR = REPO_ROOT / "paper" / "generated" / "potential"
EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"
BASE_SCORES = EVALUATION_DIR / "paper_base_auto_scores.csv"
TOOL_MANIFEST = REPO_ROOT / "results" / "manifests" / "tool_routing_image15.jsonl"
EXPERIMENTS_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260805

EXCLUDED_MODELS = {"qwen3.6-27b", "bonsai-27b"}
MODEL_ORDER = [
    "gemma-4-31b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b",
    "gemma-4-26b-a4b",
    "bonsai-8b",
    "gpt-5.6-luna-medium",
    "gpt-5.6-luna-none",
    "gpt-5.4-mini-medium",
    "gpt-5.4-mini-none",
    "gpt-5.4-nano-medium",
    "gpt-5.4-nano-none",
]
MODEL_NAMES = {
    "gpt-5.6-luna-none": "GPT-5.6-Luna (Instant)",
    "gpt-5.6-luna-medium": "GPT-5.6-Luna (Medium)",
    "gpt-5.4-mini-none": "GPT-5.4-Mini (Instant)",
    "gpt-5.4-mini-medium": "GPT-5.4-Mini (Medium)",
    "gpt-5.4-nano-none": "GPT-5.4-Nano (Instant)",
    "gpt-5.4-nano-medium": "GPT-5.4-Nano (Medium)",
    "gemma-4-31b": "Gemma4 31B",
    "gemma-4-26b-a4b": "Gemma4-26B-A4B",
    "gpt-oss-20b": "GPT-OSS-20B (Default)",
    "gpt-oss-20b-low": "GPT-OSS-20B (Low)",
    "gpt-oss-20b-medium": "GPT-OSS-20B (Medium)",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "bonsai-8b": "Bonsai-8B",
}
MODEL_FAMILIES = {
    "gpt-5.6-luna-none": "luna",
    "gpt-5.6-luna-medium": "luna",
    "gpt-5.4-mini-none": "mini",
    "gpt-5.4-mini-medium": "mini",
    "gpt-5.4-nano-none": "nano",
    "gpt-5.4-nano-medium": "nano",
    "gemma-4-31b": "gemma31",
    "gemma-4-26b-a4b": "gemma26",
    "gpt-oss-20b": "gptoss",
    "gpt-oss-20b-low": "gptoss",
    "gpt-oss-20b-medium": "gptoss",
    "qwen3.6-35b-a3b": "qwen",
    "bonsai-8b": "bonsai",
}
LENGTH_REASONS = {"length", "max_tokens", "max_output_tokens"}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def e2e_correct(df: pd.DataFrame, correct_field: str = "correct") -> pd.Series:
    correct = truthy(df[correct_field]) if correct_field in df else pd.Series(False, index=df.index)
    parsed = truthy(df["parse_success"]) if "parse_success" in df else pd.Series(True, index=df.index)
    return df["status"].astype(str).eq("success") & parsed & correct


def rate(df: pd.DataFrame, predicate: pd.Series) -> float:
    return float(predicate.reindex(df.index, fill_value=False).mean()) if len(df) else float("nan")


def success_output_rate(df: pd.DataFrame) -> float:
    success = df[df["status"].astype(str).eq("success")]
    return rate(success, e2e_correct(success))


def display_name(label: str) -> str:
    return MODEL_NAMES.get(label, label)


def model_sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {label: index for index, label in enumerate(MODEL_ORDER)}
    out = df.copy()
    out["_order"] = out["model_label"].map(order).fillna(len(order))
    return out.sort_values(["_order", "model_label"]).drop(columns="_order").reset_index(drop=True)


def load_base() -> pd.DataFrame:
    df = pd.read_csv(BASE_SCORES, low_memory=False)
    return df[~df["model_label"].astype(str).isin(EXCLUDED_MODELS)].copy()


def load_tool() -> pd.DataFrame:
    frames = []
    for label in MODEL_ORDER:
        path = EVALUATION_DIR / f"paper_tool_{label}_scores.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_case() -> pd.DataFrame:
    candidates = [
        EVALUATION_DIR / "paper_case_study_scores.csv",
        EVALUATION_DIR / "paper_case_study_api_scores.csv",
        *sorted(EVALUATION_DIR.glob("paper_case_study_api_*_scores.csv")),
    ]
    frames = [pd.read_csv(path, low_memory=False) for path in candidates if path.exists()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[~out["model_label"].astype(str).isin(EXCLUDED_MODELS)]
    return out.drop_duplicates(["model_label", "task_id"], keep="last")


def load_reliability() -> pd.DataFrame:
    frames = []
    for label in MODEL_ORDER:
        path = EVALUATION_DIR / f"paper_reliability_{label}_scores.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cluster_bootstrap(
    rows: pd.DataFrame,
    value_columns: list[str],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap model families while retaining all configurations in a sampled family."""
    clean = rows.dropna(subset=value_columns).copy()
    clean["family"] = clean["model_label"].map(MODEL_FAMILIES)
    clean = clean.dropna(subset=["family"])
    families = sorted(clean["family"].unique())
    if not families:
        empty = np.full(len(value_columns), np.nan)
        return empty, empty, empty
    arrays = {family: clean.loc[clean["family"].eq(family), value_columns].to_numpy(float) for family in families}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        selected = rng.choice(families, size=len(families), replace=True)
        sample = np.concatenate([arrays[family] for family in selected], axis=0)
        means = np.nanmean(sample, axis=0)
        samples.append(transform(means) if transform else means)
    sample_array = np.asarray(samples, dtype=float)
    point = clean[value_columns].mean().to_numpy(float)
    if transform:
        point = transform(point)
    low, high = np.quantile(sample_array, [0.025, 0.975], axis=0)
    return point, low, high


def build_direct_metrics(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = base[base["suite"].astype(str).eq("core")].copy()
    rows = []
    diagnostics = []
    for label in MODEL_ORDER:
        model_rows = core[core["model_label"].eq(label)]
        if model_rows.empty:
            continue
        metrics: dict[str, Any] = {"model_label": label, "model": display_name(label)}
        for encoding in ["raw_table", "raw_text", "processed_table", "processed_text"]:
            subset = model_rows[model_rows["encoding"].eq(encoding)]
            metrics[f"{encoding}_success_output_accuracy"] = success_output_rate(subset)
            metrics[f"{encoding}_e2e_accuracy"] = rate(subset, e2e_correct(subset))
            metrics[f"{encoding}_errors"] = int((~subset["status"].astype(str).eq("success")).sum())
            success = subset[subset["status"].astype(str).eq("success")]
            metrics[f"{encoding}_parse_failures"] = int((~truthy(success["parse_success"])).sum())
        raw = model_rows[model_rows["encoding"].isin(["raw_table", "raw_text"])]
        processed = model_rows[model_rows["encoding"].isin(["processed_table", "processed_text"])]
        metrics["raw_success_output_accuracy"] = success_output_rate(raw)
        metrics["raw_e2e_accuracy"] = rate(raw, e2e_correct(raw))
        metrics["processed_success_output_accuracy"] = success_output_rate(processed)
        metrics["processed_e2e_accuracy"] = rate(processed, e2e_correct(processed))
        metrics["raw_e2e_drop"] = metrics["processed_e2e_accuracy"] - metrics["raw_e2e_accuracy"]
        rows.append(metrics)

        raw_success = raw[raw["status"].astype(str).eq("success")]
        diagnostics.append(
            {
                "model_label": label,
                "model": display_name(label),
                "raw_expected": len(raw),
                "raw_errors": int((~raw["status"].astype(str).eq("success")).sum()),
                "raw_parse_failures": int((~truthy(raw_success["parse_success"])).sum()),
                "raw_empty_outputs": int(raw["response"].fillna("").astype(str).str.strip().eq("").sum()),
                "raw_length_stops": count_length_stops(raw),
                "raw_success_output_accuracy": success_output_rate(raw),
                "raw_e2e_accuracy": rate(raw, e2e_correct(raw)),
                "processed_e2e_accuracy": rate(processed, e2e_correct(processed)),
            }
        )
    return model_sort(pd.DataFrame(rows)), model_sort(pd.DataFrame(diagnostics))


def count_length_stops(df: pd.DataFrame) -> int:
    count = 0
    for row in df.itertuples(index=False):
        model_label = str(getattr(row, "model_label", ""))
        run_id = str(getattr(row, "source_run_id", ""))
        task_id = str(getattr(row, "task_id", ""))
        path = REPO_ROOT / "results" / "raw" / slugify(model_label) / slugify(run_id) / f"{task_id}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("finish_reason", "")).strip().lower() in LENGTH_REASONS:
            count += 1
    return count


def build_threshold_sensitivity(base: pd.DataFrame) -> pd.DataFrame:
    core = base[base["suite"].astype(str).eq("core")].copy()
    groups = {
        "Raw average": ["raw_table", "raw_text"],
        "Raw table": ["raw_table"],
        "Raw text": ["raw_text"],
        "Processed average": ["processed_table", "processed_text"],
        "Processed table": ["processed_table"],
        "Processed text": ["processed_text"],
    }
    thresholds = ["0.1", "0.5", "1", "5"]
    rows = []
    for label, encodings in groups.items():
        subset = core[core["encoding"].isin(encodings)]
        model_rows = []
        for model, model_df in subset.groupby("model_label"):
            result: dict[str, Any] = {"model_label": model}
            for threshold in thresholds:
                field = f"correct_within_{threshold}pct"
                result[f"within_{threshold}pct"] = float(truthy(model_df[field]).mean())
            model_rows.append(result)
        model_summary = pd.DataFrame(model_rows)
        values = [f"within_{threshold}pct" for threshold in thresholds]
        point, low, high = cluster_bootstrap(model_summary, values)
        for index, threshold in enumerate(thresholds):
            rows.append(
                {
                    "input_format": label,
                    "threshold_pct": float(threshold),
                    "accuracy": point[index],
                    "ci95_low": low[index],
                    "ci95_high": high[index],
                    "bootstrap_unit": "model family",
                }
            )
    return pd.DataFrame(rows)


def build_direct_tolerance_table(base: pd.DataFrame) -> pd.DataFrame:
    core = base[base["suite"].astype(str).eq("core")].copy()
    encodings = [
        ("Raw text", "raw_text"),
        ("Raw tabular", "raw_table"),
        ("Processed text", "processed_text"),
        ("Processed tabular", "processed_table"),
    ]
    thresholds = ["0.1", "0.5", "1", "5"]
    rows = []
    for label, encoding in encodings:
        subset = core[core["encoding"].eq(encoding)]
        row: dict[str, Any] = {"direct_input": label, "encoding": encoding}
        for threshold in thresholds:
            field = f"correct_within_{threshold}pct"
            model_rates = []
            for _, model_df in subset.groupby("model_label"):
                correct = (
                    model_df["status"].astype(str).eq("success")
                    & truthy(model_df["parse_success"])
                    & truthy(model_df[field])
                )
                model_rates.append(float(correct.mean()))
            row[f"within_{threshold}pct"] = float(np.mean(model_rates)) if model_rates else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_statistic_representation_table(base: pd.DataFrame) -> pd.DataFrame:
    core = base[base["suite"].astype(str).eq("core")].copy()
    statistic_order = [
        ("Mean", "mean"),
        ("Median", "median"),
        ("Range", "range"),
        ("Minimum", "min"),
        ("Maximum", "max"),
        ("Q1", "q1"),
        ("Q3", "q3"),
        ("IQR", "iqr"),
    ]
    encodings = [
        ("raw_text", "Raw text"),
        ("raw_table", "Raw tabular"),
        ("processed_text", "Processed text"),
        ("processed_table", "Processed tabular"),
    ]
    rows = []
    for statistic_label, statistic in statistic_order:
        row: dict[str, Any] = {"statistic": statistic, "Statistic": statistic_label}
        for encoding, column_label in encodings:
            subset = core[
                core["statistic"].astype(str).eq(statistic)
                & core["encoding"].astype(str).eq(encoding)
            ]
            model_rates = []
            for _, model_df in subset.groupby("model_label"):
                correct = (
                    model_df["status"].astype(str).eq("success")
                    & truthy(model_df["parse_success"])
                    & truthy(model_df["correct_within_1pct"])
                )
                model_rates.append(float(correct.mean()))
            row[column_label] = float(np.mean(model_rates)) if model_rates else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_raw_statistic_diagnostics(base: pd.DataFrame) -> pd.DataFrame:
    raw = base[
        base["suite"].astype(str).eq("core")
        & base["encoding"].isin(["raw_table", "raw_text"])
    ].copy()
    rows = []
    for (model, statistic), group in raw.groupby(["model_label", "statistic"]):
        rows.append(
            {
                "model_label": model,
                "statistic": statistic,
                "accuracy": rate(group, e2e_correct(group)),
                "errors": int((~group["status"].astype(str).eq("success")).sum()),
            }
        )
    detail = pd.DataFrame(rows)
    output = []
    for statistic, group in detail.groupby("statistic"):
        point, low, high = cluster_bootstrap(group, ["accuracy"])
        output.append(
            {
                "statistic": statistic,
                "model_macro_accuracy": point[0],
                "ci95_low": low[0],
                "ci95_high": high[0],
                "non_successful_records": int(group["errors"].sum()),
            }
        )
    order = {name: index for index, name in enumerate(["max", "min", "range", "mean", "median", "q3", "q1", "iqr"])}
    return pd.DataFrame(output).sort_values("statistic", key=lambda s: s.map(order)).reset_index(drop=True)


def model_level_tool_metrics(tool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in tool.groupby("model_label"):
        accepted = (
            group["status"].astype(str).eq("success")
            & truthy(group["tool_routing_correct"])
            & truthy(group["tool_result_correct"])
        )
        strict = (
            group["status"].astype(str).eq("success")
            & truthy(group["tool_routing_strict_correct"])
            & truthy(group["tool_result_correct"])
        )
        executable = group["status"].astype(str).eq("success") & truthy(group["tool_executable"])
        rows.append(
            {
                "model_label": label,
                "accepted_e2e": float(accepted.mean()),
                "strict_e2e": float(strict.mean()),
                "executable_e2e": float(executable.mean()),
            }
        )
    return model_sort(pd.DataFrame(rows))


def model_level_case_metrics(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in case.groupby("model_label"):
        numeric = group[
            group["case_study_stage"].astype(str).eq("1") & group["answer_type"].astype(str).eq("numeric")
        ]
        pairwise = group[
            group["case_study_stage"].astype(str).eq("1") & group["answer_type"].astype(str).eq("label")
        ]
        topk = group[
            group["case_study_stage"].astype(str).eq("2") & group["answer_type"].astype(str).eq("topk_metrics")
        ]
        exact_values = pd.to_numeric(topk["topk_precision_at_3"], errors="coerce")
        family_values = pd.to_numeric(topk["topk_family_precision_at_3"], errors="coerce")
        rows.append(
            {
                "model_label": label,
                "numeric_e2e": rate(numeric, e2e_correct(numeric)),
                "pairwise_e2e": rate(pairwise, e2e_correct(pairwise)),
                "top3_parse_rate": exact_values.notna().mean(),
                "exact_top3_conditional": exact_values.mean(),
                "exact_top3_e2e": exact_values.fillna(0).mean(),
                "family_top3_conditional": family_values.mean(),
                "family_top3_e2e": family_values.fillna(0).mean(),
                "exact_set_match_e2e": truthy(topk["topk_exact_set_match"]).mean(),
                "valid_metric_rate_e2e": pd.to_numeric(topk["topk_valid_metric_rate"], errors="coerce").fillna(0).mean(),
            }
        )
    return model_sort(pd.DataFrame(rows))


def model_level_reliability_metrics(reliability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in reliability.groupby("model_label"):
        parsed = group[
            group["status"].astype(str).eq("success") & truthy(group["parse_success"])
        ]
        rows.append(
            {
                "model_label": label,
                "model": display_name(label),
                "repeat_parse_rate": len(parsed) / len(group),
                "repeat_parsed_accuracy": truthy(parsed["correct"]).mean() if len(parsed) else float("nan"),
                "repeat_accuracy_e2e": rate(group, e2e_correct(group)),
                "total": len(group),
                "non_success": int((~group["status"].astype(str).eq("success")).sum()),
                "parse_failures": int(
                    (~truthy(group.loc[group["status"].astype(str).eq("success"), "parse_success"])).sum()
                ),
            }
        )
    return model_sort(pd.DataFrame(rows))


def build_claim_intervals(
    direct: pd.DataFrame,
    tool: pd.DataFrame,
    case: pd.DataFrame,
) -> pd.DataFrame:
    merged = direct[["model_label", "processed_e2e_accuracy", "raw_e2e_accuracy"]].merge(
        tool[["model_label", "accepted_e2e"]], on="model_label", how="inner"
    ).merge(
        case[["model_label", "pairwise_e2e", "family_top3_e2e", "exact_top3_e2e"]],
        on="model_label",
        how="inner",
    )
    specs = [
        ("Processed direct - raw direct", "processed_e2e_accuracy", "raw_e2e_accuracy"),
        ("Accepted tool call - raw direct", "accepted_e2e", "raw_e2e_accuracy"),
        ("Pairwise - family top-3", "pairwise_e2e", "family_top3_e2e"),
        ("Pairwise - exact top-3", "pairwise_e2e", "exact_top3_e2e"),
    ]
    rows = []
    for claim, first, second in specs:
        point, low, high = cluster_bootstrap(
            merged,
            [first, second],
            transform=lambda values: np.asarray([values[0], values[1], values[0] - values[1]]),
        )
        rows.append(
            {
                "comparison": claim,
                "first_mean": point[0],
                "first_ci95_low": low[0],
                "first_ci95_high": high[0],
                "second_mean": point[1],
                "second_ci95_low": low[1],
                "second_ci95_high": high[1],
                "paired_difference": point[2],
                "difference_ci95_low": low[2],
                "difference_ci95_high": high[2],
                "models": len(merged),
                "families": merged["model_label"].map(MODEL_FAMILIES).nunique(),
            }
        )
    return pd.DataFrame(rows)


def normalized_request(task: dict[str, Any]) -> str:
    text = str(task.get("user_request", "")).lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def infer_tool_type(task: dict[str, Any], request: str) -> str:
    if re.search(r"\btop\s*3\b|\btop three\b|\bwhich three\b|\bthree characterization metrics\b", request):
        return "top_morphology_differences"
    if task.get("image_a_id") not in (None, "") and task.get("image_b_id") not in (None, ""):
        return "compare_morphology_statistic"
    return "get_morphology_statistic"


def literal_route(task: dict[str, Any]) -> dict[str, Any] | None:
    request = normalized_request(task)
    tool = infer_tool_type(task, request)
    if tool == "top_morphology_differences":
        return {
            "tool": tool,
            "arguments": {
                "image_a_id": int(task["image_a_id"]),
                "image_b_id": int(task["image_b_id"]),
                "top_k": 3,
                "metric_scope": "expanded_morphology",
            },
        }
    feature = None
    if re.search(r"\bequivalent[ -]diameter\b", request):
        feature = "equivalent_diameter"
    elif "circularity" in request:
        feature = "circularity"
    elif "aspect ratio" in request:
        feature = "aspect_ratio"
    elif "elongation" in request:
        feature = "elongation"
    statistic = None
    if re.search(r"\bd10\b|10th percentile|percentile 10", request):
        statistic = "percentile_10"
    elif re.search(r"\bd50\b|50th percentile|percentile 50", request):
        statistic = "percentile_50"
    elif re.search(r"\bd90\b|90th percentile|percentile 90", request):
        statistic = "percentile_90"
    elif "median" in request:
        statistic = "median"
    elif "d90 minus d10 over d50" in request or "d90-d10" in request:
        statistic = "d90_minus_d10_over_d50"
    if feature is None or statistic is None:
        return None
    if tool == "get_morphology_statistic":
        arguments = {"image_id": int(task["image_id"]), "feature": feature, "statistic": statistic}
    else:
        relation = "lower" if re.search(r"\blower\b", request) else "greater"
        arguments = {
            "image_a_id": int(task["image_a_id"]),
            "image_b_id": int(task["image_b_id"]),
            "feature": feature,
            "statistic": statistic,
            "relation": relation,
        }
    return {"tool": tool, "arguments": arguments}


def mapped_keyword_route(task: dict[str, Any]) -> dict[str, Any] | None:
    request = normalized_request(task)
    tool = infer_tool_type(task, request)
    if tool == "top_morphology_differences":
        return {
            "tool": tool,
            "arguments": {
                "image_a_id": int(task["image_a_id"]),
                "image_b_id": int(task["image_b_id"]),
                "top_k": 3,
                "metric_scope": "expanded_morphology",
            },
        }
    feature = None
    statistic = None
    if re.search(r"round|spheric|circular", request):
        feature, statistic = "circularity", "median"
    elif re.search(r"elongat|needle-like|needle like", request):
        feature = "elongation" if "elongation" in request else "aspect_ratio"
        statistic = "median"
    elif re.search(r"spread|width|wider|broad|distribution|polydisperse|span", request):
        feature, statistic = "equivalent_diameter", "d90_minus_d10_over_d50"
    elif re.search(r"\bd10\b|fine tail|small particle", request):
        feature, statistic = "equivalent_diameter", "percentile_10"
    elif re.search(r"\bd50\b|typical|middle particle", request):
        feature, statistic = "equivalent_diameter", "percentile_50"
    elif re.search(r"\bd90\b|bigger chunks|coarse|top tail|largest particle", request):
        feature, statistic = "equivalent_diameter", "percentile_90"
    if feature is None or statistic is None:
        return literal_route(task)
    if tool == "get_morphology_statistic":
        arguments = {"image_id": int(task["image_id"]), "feature": feature, "statistic": statistic}
    else:
        relation = "lower" if re.search(r"fewer|less|lower", request) else "greater"
        arguments = {
            "image_a_id": int(task["image_a_id"]),
            "image_b_id": int(task["image_b_id"]),
            "feature": feature,
            "statistic": statistic,
            "relation": relation,
        }
    return {"tool": tool, "arguments": arguments}


def build_routing_baselines(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = list(iter_jsonl(TOOL_MANIFEST))
    baseline_functions = [
        ("Literal schema terms", literal_route),
        ("Mapped keyword rules", mapped_keyword_route),
    ]
    scored_rows = []
    for baseline, route_function in baseline_functions:
        for task in tasks:
            call = route_function(task)
            response = json.dumps(call, separators=(",", ":")) if call is not None else "{}"
            score = score_tool_call(task, response, config)
            scored_rows.append(
                {
                    "baseline": baseline,
                    "task_id": task["task_id"],
                    "tool_task_type": task["tool_task_type"],
                    "paraphrase_family": task["paraphrase_family"],
                    "predicted_call": response,
                    **score,
                }
            )
    scores = pd.DataFrame(scored_rows)
    summaries = []
    for (baseline, task_type), group in scores.groupby(["baseline", "tool_task_type"], dropna=False):
        summaries.append(summarize_baseline_group(baseline, str(task_type), group))
    for baseline, group in scores.groupby("baseline"):
        summaries.append(summarize_baseline_group(baseline, "all", group))
    return scores, pd.DataFrame(summaries)


def summarize_baseline_group(baseline: str, task_type: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "task_type": task_type,
        "tasks": len(group),
        "valid_tool_call": truthy(group["tool_call_valid"]).mean(),
        "executable": truthy(group["tool_executable"]).mean(),
        "accepted_route": truthy(group["tool_routing_correct"]).mean(),
        "strict_route": truthy(group["tool_routing_strict_correct"]).mean(),
        "correct_result": truthy(group["tool_result_correct"]).mean(),
    }


def compact_call(value: Any) -> str:
    try:
        payload = json.loads(str(value)) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return str(value)
    if not isinstance(payload, dict):
        return str(payload)
    tool = payload.get("tool", "")
    args = payload.get("arguments", payload)
    return f"{tool}({', '.join(f'{key}={val}' for key, val in args.items())})"


def build_accepted_strict_examples(tool: pd.DataFrame) -> pd.DataFrame:
    tasks = {task["task_id"]: task for task in iter_jsonl(TOOL_MANIFEST)}
    candidates = tool[
        truthy(tool["tool_routing_correct"])
        & ~truthy(tool["tool_routing_strict_correct"])
    ].copy()
    preferences = ["fewer_elongated", "typical_size_d50", "roundness_median"]
    selected = []
    seen_reasons: set[str] = set()
    for intent in preferences:
        subset = candidates[candidates["tool_intent_key"].astype(str).eq(intent)]
        for _, row in subset.iterrows():
            task = tasks.get(str(row["task_id"]))
            if not task:
                continue
            emitted = json.loads(row["tool_predicted_arguments"])
            expected = task.get("expected_arguments", {})
            if emitted.get("statistic") != expected.get("statistic"):
                reason = "median and percentile_50 return the same deterministic statistic"
            elif emitted.get("feature") != expected.get("feature"):
                reason = "aspect_ratio and elongation preserve this deterministic comparison"
            else:
                reason = "normalized alias preserves the deterministic result"
            if reason in seen_reasons:
                continue
            seen_reasons.add(reason)
            selected.append(
                {
                    "model": display_name(str(row["model_label"])),
                    "request": task.get("user_request", ""),
                    "canonical_call": compact_call(
                        {"tool": task.get("expected_tool"), "arguments": expected}
                    ),
                    "emitted_call": compact_call(
                        {"tool": row.get("tool_predicted_tool"), "arguments": emitted}
                    ),
                    "why_accepted": reason,
                }
            )
            break
        if len(selected) >= 2:
            break
    return pd.DataFrame(selected)


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def pct(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "--" if not np.isfinite(number) else f"{float(number) * 100:.1f}"


def ci_text(mean: Any, low: Any, high: Any) -> str:
    return f"{pct(mean)} [{pct(low)}, {pct(high)}]"


def pct_with_symbol(value: Any, *, bold: bool = False, star: bool = False) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(number):
        return "--"
    rendered = f"{float(number) * 100:.1f}\\%"
    if bold:
        rendered = f"\\textbf{{{rendered}}}"
    if star:
        rendered = f"{rendered}\\textsuperscript{{*}}"
    return rendered


def rounded_pct(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float("nan") if not np.isfinite(number) else round(float(number) * 100, 1)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_end_to_end_table(diagnostics: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{Direct-answer denominator audit. Successful-output accuracy omits non-successful requests; end-to-end accuracy counts errors, empty responses, parse failures, and length stops as incorrect. Values are percentages.}",
        r"\label{tab:denominator-audit}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & \shortstack{Raw success-\\output acc.} & \shortstack{Raw end-\\to-end acc.} & Errors & Unparsed & Length & \shortstack{Proc. end-\\to-end acc.} \\",
        r"\midrule",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.model)} & {pct(row.raw_success_output_accuracy)} & "
            f"{pct(row.raw_e2e_accuracy)} & {row.raw_errors} & {row.raw_parse_failures} & "
            f"{row.raw_length_stops} & {pct(row.processed_e2e_accuracy)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_end_to_end_audit.tex", "\n".join(lines))


def write_threshold_table(thresholds: pd.DataFrame) -> None:
    pivot = thresholds.pivot(index="input_format", columns="threshold_pct", values="accuracy")
    order = ["Raw table", "Raw text", "Raw average", "Processed table", "Processed text", "Processed average"]
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{End-to-end direct accuracy under numeric tolerance thresholds. Non-successful and unparsed outputs count as incorrect. Values are model-macro percentages.}",
        r"\label{tab:threshold-sensitivity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Input format & 0.1\% & 0.5\% & 1\% & 5\% \\",
        r"\midrule",
    ]
    for label in order:
        if label not in pivot.index:
            continue
        values = pivot.loc[label]
        lines.append(
            f"{tex_escape(label)} & {pct(values.get(0.1))} & {pct(values.get(0.5))} & "
            f"{pct(values.get(1.0))} & {pct(values.get(5.0))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_threshold_sensitivity.tex", "\n".join(lines))


def write_direct_tolerance_table(tolerance: pd.DataFrame) -> None:
    threshold_columns = ["within_0.1pct", "within_0.5pct", "within_1pct", "within_5pct"]
    best_by_column = {
        column: tolerance[column].map(rounded_pct).max(skipna=True)
        for column in threshold_columns
    }
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Direct-inference sensitivity to numeric tolerance.} Accuracy across raw and processed text and tabular representations under increasingly relaxed numeric-error thresholds. Best performance at each threshold is \textbf{bolded}.}",
        r"\label{tab:tolerance-sensitivity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Direct Input & 0.1\% & 0.5\% & 1\% & 5\% \\",
        r"\midrule",
    ]
    for _, row in tolerance.iterrows():
        values = []
        for column in threshold_columns:
            value = row[column]
            values.append(
                pct_with_symbol(
                    value,
                    bold=rounded_pct(value) == best_by_column[column],
                )
            )
        lines.append(f"{tex_escape(row['direct_input'])} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_direct_tolerance_sensitivity.tex", "\n".join(lines))


def write_statistic_representation_table(statistics: pd.DataFrame) -> None:
    representation_columns = ["Raw text", "Raw tabular", "Processed text", "Processed tabular"]
    best_by_column = {
        column: statistics[column].map(rounded_pct).max(skipna=True)
        for column in representation_columns
    }
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Direct-inference accuracy by requested statistic.} Accuracy at the 1\% numeric-error threshold across raw and processed text and tabular representations. The highest value within each representation is \textbf{bolded}. The best representation for a given statistic is marked by $^{*}$.}",
        r"\label{tab:statistic-performance}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Statistic & Raw text & Raw tabular & Processed text & Processed tabular \\",
        r"\midrule",
    ]
    for _, row in statistics.iterrows():
        row_values = [row[column] for column in representation_columns]
        rounded_values = [rounded_pct(value) for value in row_values]
        finite_values = [value for value in rounded_values if np.isfinite(value)]
        best_in_row = max(finite_values) if finite_values else float("nan")
        cells = []
        for column, value, rounded_value in zip(representation_columns, row_values, rounded_values):
            cells.append(
                pct_with_symbol(
                    value,
                    bold=rounded_value == best_by_column[column],
                    star=rounded_value == best_in_row,
                )
            )
        lines.append(f"{tex_escape(row['Statistic'])} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_statistic_performance.tex", "\n".join(lines))


def write_claim_ci_table(claims: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Architecture-family cluster-bootstrap intervals for the principal paired comparisons. Values are percentage points with 95\% intervals in brackets.}",
        r"\label{tab:claim-confidence}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & First & Second & Difference \\",
        r"\midrule",
    ]
    for row in claims.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.comparison)} & "
            f"{ci_text(row.first_mean, row.first_ci95_low, row.first_ci95_high)} & "
            f"{ci_text(row.second_mean, row.second_ci95_low, row.second_ci95_high)} & "
            f"{ci_text(row.paired_difference, row.difference_ci95_low, row.difference_ci95_high)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_claim_confidence_intervals.tex", "\n".join(lines))


def write_raw_statistic_table(stats: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Raw-record end-to-end accuracy by requested statistic. Intervals use architecture-family cluster bootstrap resampling.}",
        r"\label{tab:raw-statistic-diagnostics}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Statistic & Accuracy [95\% CI] & Non-successful records \\",
        r"\midrule",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"{tex_escape(str(row.statistic).upper())} & "
            f"{ci_text(row.model_macro_accuracy, row.ci95_low, row.ci95_high)} & "
            f"{row.non_successful_records} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_raw_statistic_diagnostics.tex", "\n".join(lines))


def write_baseline_table(summary: pd.DataFrame) -> None:
    all_rows = summary[summary["task_type"].eq("all")]
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Non-LLM routing baselines over the 555 schema-guided prompts. The mapped baseline implements the same explicit jargon map supplied to the LLMs.}",
        r"\label{tab:routing-baselines}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Router & Valid call & Executable & Accepted & Strict \\",
        r"\midrule",
    ]
    for row in all_rows.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.baseline)} & {pct(row.valid_tool_call)} & {pct(row.executable)} & "
            f"{pct(row.accepted_route)} & {pct(row.strict_route)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_routing_baselines.tex", "\n".join(lines))


def write_accepted_examples_table(examples: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Accepted versus strict tool-call examples. Accepted alternatives are retained only when deterministic execution preserves the benchmark answer.}",
        r"\label{tab:accepted-strict-examples}",
        r"\begin{tabular}{p{0.16\linewidth}p{0.23\linewidth}p{0.23\linewidth}p{0.29\linewidth}}",
        r"\toprule",
        r"Model & Canonical call & Emitted call & Why accepted \\",
        r"\midrule",
    ]
    for row in examples.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.model)} & \\texttt{{{tex_escape(row.canonical_call)}}} & "
            f"\\texttt{{{tex_escape(row.emitted_call)}}} & {tex_escape(row.why_accepted)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_accepted_strict_examples.tex", "\n".join(lines))


def write_side_denominator_table(reliability: pd.DataFrame, case: pd.DataFrame) -> None:
    merged = reliability.merge(case, on="model_label", how="outer", suffixes=("", "_case"))
    merged["model"] = merged["model_label"].map(display_name)
    merged = model_sort(merged)
    lines = [
        "% Candidate appendix table; generated by src/build_potential_analyses.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\caption{Denominator sensitivity in side studies. Conditional values omit unparsed answers; end-to-end values retain every expected prompt and score unparsed or non-successful outputs as zero. Values are percentages.}",
        r"\label{tab:side-denominator-audit}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Repeated direct} & \multicolumn{3}{c}{Family top-3} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Model & Parsed & End-to-end & Parse rate & Conditional & End-to-end & Parse rate \\",
        r"\midrule",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.model)} & {pct(row.repeat_parsed_accuracy)} & "
            f"{pct(row.repeat_accuracy_e2e)} & {pct(row.repeat_parse_rate)} & "
            f"{pct(row.family_top3_conditional)} & {pct(row.family_top3_e2e)} & "
            f"{pct(row.top3_parse_rate)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(OUTPUT_DIR / "table_side_study_denominator_audit.tex", "\n".join(lines))


def write_assessment(
    direct: pd.DataFrame,
    claims: pd.DataFrame,
    thresholds: pd.DataFrame,
    stats: pd.DataFrame,
    baselines: pd.DataFrame,
    reliability: pd.DataFrame,
    case: pd.DataFrame,
) -> None:
    processed_raw = claims.iloc[0]
    route_raw = claims.iloc[1]
    pair_exact = claims.iloc[3]
    raw_thresholds = thresholds[thresholds["input_format"].eq("Raw average")].set_index("threshold_pct")
    mapped = baselines[(baselines["baseline"].eq("Mapped keyword rules")) & baselines["task_type"].eq("all")].iloc[0]
    literal = baselines[(baselines["baseline"].eq("Literal schema terms")) & baselines["task_type"].eq("all")].iloc[0]
    qwen = direct[direct["model_label"].eq("qwen3.6-35b-a3b")].iloc[0]
    mini_repeat = reliability[reliability["model_label"].eq("gpt-5.4-mini-medium")].iloc[0]
    mini_case = case[case["model_label"].eq("gpt-5.4-mini-medium")].iloc[0]
    text = f"""# Potential Reviewer-Facing Analyses

These artifacts are exploratory and do not modify the main paper table or figure.

## Recommended Promotions

1. **Use end-to-end direct accuracy in the paper.** Processed direct averages {processed_raw.first_mean * 100:.1f}% and raw direct averages {processed_raw.second_mean * 100:.1f}% when every expected request remains in the denominator. The paired gap is {processed_raw.paired_difference * 100:.1f} percentage points (architecture-family cluster-bootstrap 95% CI: {processed_raw.difference_ci95_low * 100:.1f} to {processed_raw.difference_ci95_high * 100:.1f}).
2. **Add threshold sensitivity to the appendix.** Raw accuracy rises from {raw_thresholds.loc[0.1, 'accuracy'] * 100:.1f}% at 0.1% tolerance to only {raw_thresholds.loc[5.0, 'accuracy'] * 100:.1f}% at 5% tolerance. The raw-record failure is therefore not an artifact of the 1% cutoff.
3. **Add one accepted-versus-strict example.** The generated example table shows that accepted scoring permits only alternatives whose deterministic execution reproduces the target result.
4. **Add the raw-statistic diagnostic.** {stats.iloc[0].statistic.upper()} and {stats.iloc[1].statistic.upper()} remain much easier than quartiles and IQR, making the arithmetic failure interpretable rather than merely aggregate.
5. **Report confidence intervals in the appendix or Results text.** Accepted tool calling exceeds raw direct accuracy by {route_raw.paired_difference * 100:.1f} points (95% CI: {route_raw.difference_ci95_low * 100:.1f} to {route_raw.difference_ci95_high * 100:.1f}). Pairwise comparison exceeds exact top-3 precision by {pair_exact.paired_difference * 100:.1f} points (95% CI: {pair_exact.difference_ci95_low * 100:.1f} to {pair_exact.difference_ci95_high * 100:.1f}).

## Denominator Finding

The current successful-output denominator is optimistic for incomplete open-weight raw runs. For example, Qwen raw accuracy changes from {qwen.raw_success_output_accuracy * 100:.1f}% over successful outputs to {qwen.raw_e2e_accuracy * 100:.1f}% end-to-end. This strengthens the representation-sensitivity claim, but Table 1 and its caption should use one denominator consistently.

The side studies have a larger conditional-scoring effect. GPT-5.4-Mini Medium repeat accuracy changes from {mini_repeat.repeat_parsed_accuracy * 100:.1f}% over parsed repetitions to {mini_repeat.repeat_accuracy_e2e * 100:.1f}% over all repetitions. Its family top-3 precision changes from {mini_case.family_top3_conditional * 100:.1f}% over parsed top-k responses to {mini_case.family_top3_e2e * 100:.1f}% over all 30 tasks. Table 1 and Figure 2 currently use the conditional case-study summary, so they should be refreshed with end-to-end side-study metrics before submission.

## Non-LLM Baseline Finding

The literal schema-term baseline reaches {literal.accepted_route * 100:.1f}% accepted routing. The hand-authored baseline using the same jargon map supplied in the prompts reaches {mapped.accepted_route * 100:.1f}% accepted and {mapped.strict_route * 100:.1f}% strict routing.

This is important and should not be hidden: the present routing benchmark is a schema-following and intent-normalization evaluation, not evidence that an LLM is uniquely necessary. A strong paper can frame this positively: narrow, stable deployments may use deterministic rules, while capable LLMs provide a configurable interface without implementing a separate parser for each vocabulary. To claim broader language generalization, a future experiment would need held-out paraphrase families or user-authored queries that were not represented in the supplied jargon map.

## Statistical Scope

Confidence intervals resample eight architecture families rather than treating individual prompts as independent. This is deliberately conservative because prompts share images, features, templates, and model families. The point estimates remain macro-averages over the 13 tested configurations.

## Files

- `direct_end_to_end_metrics.csv`: per-model direct accuracy under both denominators.
- `failure_diagnostics.csv`: errors, parse failures, empty outputs, and length stops.
- `threshold_sensitivity.csv`: 0.1%, 0.5%, 1%, and 5% tolerance results.
- `claim_confidence_intervals.csv`: paired headline comparisons with cluster-bootstrap intervals.
- `raw_statistic_diagnostics.csv`: raw performance by requested statistic.
- `routing_baseline_summary.csv`: literal and mapped non-LLM routers.
- `table_side_study_denominator_audit.tex`: conditional versus end-to-end side-study values.
- `accepted_strict_examples.csv`: concrete scientifically equivalent routes.
- Candidate LaTeX tables mirror these CSV files.
"""
    write_text(OUTPUT_DIR / "README.md", text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_yaml(EXPERIMENTS_CONFIG)
    base = load_base()
    tool_scores = load_tool()
    case_scores = load_case()
    reliability_scores = load_reliability()

    direct, diagnostics = build_direct_metrics(base)
    thresholds = build_threshold_sensitivity(base)
    direct_tolerance = build_direct_tolerance_table(base)
    statistic_representations = build_statistic_representation_table(base)
    raw_stats = build_raw_statistic_diagnostics(base)
    tool = model_level_tool_metrics(tool_scores)
    case = model_level_case_metrics(case_scores)
    reliability = model_level_reliability_metrics(reliability_scores)
    claims = build_claim_intervals(direct, tool, case)
    baseline_scores, baseline_summary = build_routing_baselines(config)
    examples = build_accepted_strict_examples(tool_scores)

    outputs = {
        "direct_end_to_end_metrics.csv": direct,
        "failure_diagnostics.csv": diagnostics,
        "threshold_sensitivity.csv": thresholds,
        "direct_tolerance_sensitivity.csv": direct_tolerance,
        "statistic_performance.csv": statistic_representations,
        "raw_statistic_diagnostics.csv": raw_stats,
        "tool_end_to_end_metrics.csv": tool,
        "case_study_end_to_end_metrics.csv": case,
        "reliability_end_to_end_metrics.csv": reliability,
        "claim_confidence_intervals.csv": claims,
        "routing_baseline_scores.csv": baseline_scores,
        "routing_baseline_summary.csv": baseline_summary,
        "accepted_strict_examples.csv": examples,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / name, index=False)

    write_end_to_end_table(diagnostics)
    write_threshold_table(thresholds)
    write_direct_tolerance_table(direct_tolerance)
    write_statistic_representation_table(statistic_representations)
    write_claim_ci_table(claims)
    write_raw_statistic_table(raw_stats)
    write_baseline_table(baseline_summary)
    write_accepted_examples_table(examples)
    write_side_denominator_table(reliability, case)
    write_assessment(direct, claims, thresholds, raw_stats, baseline_summary, reliability, case)
    print(f"Wrote {len(outputs) + 10} potential artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
