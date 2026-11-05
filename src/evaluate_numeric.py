#!/usr/bin/env python3
"""Evaluate AI4Mat model outputs against deterministic ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ai4mat_common import REPO_ROOT, iter_jsonl, load_yaml, slugify, write_json_atomic
import tool_routing as tr


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
DEFAULT_RUN_ID_CANDIDATES = ["run1", "vllm-run1"]
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=False)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--model-label", action="append", default=None)
    parser.add_argument(
        "--run-id",
        default="auto",
        help=(
            "Run folder to evaluate. Use 'auto' to choose the fullest run per model "
            "from --run-id-candidate values. Default: auto."
        ),
    )
    parser.add_argument(
        "--run-id-candidate",
        action="append",
        default=None,
        help=(
            "Candidate run folder for --run-id auto. Can be repeated. "
            "Default candidates: run1, vllm-run1."
        ),
    )
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--numeric-parser",
        choices=["first", "final_clean"],
        default="first",
        help=(
            "Numeric extraction mode. 'first' is the strict benchmark parser; "
            "'final_clean' removes known turn/thinking artifacts and uses the final number."
        ),
    )
    return parser.parse_args()


def parse_number(text: str) -> float | None:
    match = NUMBER_RE.search((text or "").strip())
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in NUMBER_RE.finditer((text or "").strip()):
        try:
            numbers.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return numbers


def clean_final_numeric_text(text: str) -> str:
    cleaned = text or ""
    if "<end_of_turn>" in cleaned:
        cleaned = cleaned.split("<end_of_turn>", 1)[0]
    if "</think>" in cleaned:
        tail = cleaned.rsplit("</think>", 1)[1]
        if NUMBER_RE.search(tail):
            cleaned = tail
    lowered = cleaned.lower()
    marker_index = max(
        lowered.rfind("[output]"),
        lowered.rfind("output:"),
        lowered.rfind("answer:"),
        lowered.rfind("final answer:"),
    )
    if marker_index != -1 and NUMBER_RE.search(cleaned[marker_index:]):
        cleaned = cleaned[marker_index:]
    return cleaned


def parse_number_with_mode(text: str, mode: str) -> float | None:
    if mode == "first":
        return parse_number(text)
    if mode == "final_clean":
        numbers = parse_numbers(clean_final_numeric_text(text))
        return numbers[-1] if numbers else None
    raise ValueError(f"Unknown numeric parser mode: {mode}")


def parse_label(text: str, valid_labels: list[str]) -> str | None:
    cleaned = clean_final_label_text(text or "")
    if not cleaned.strip():
        return None

    stripped = cleaned.strip().strip('"').strip("'").strip(" .,:;")
    lower_labels = {label.lower(): label for label in valid_labels}
    if stripped.lower() in lower_labels:
        return lower_labels[stripped.lower()]

    candidates: list[tuple[int, str]] = []
    if "tie" in valid_labels:
        candidates.extend((match.start(), "tie") for match in re.finditer(r"\btie\b", cleaned, re.IGNORECASE))
    if "A" in valid_labels:
        patterns = [
            r"\b(?:image|sample|powder|batch)\s+A\b",
            r"(?<![A-Za-z0-9])A(?![A-Za-z0-9])",
        ]
        for pattern in patterns:
            candidates.extend((match.start(), "A") for match in re.finditer(pattern, cleaned))
    if "B" in valid_labels:
        patterns = [
            r"\b(?:image|sample|powder|batch)\s+B\b",
            r"(?<![A-Za-z0-9])B(?![A-Za-z0-9])",
        ]
        for pattern in patterns:
            candidates.extend((match.start(), "B") for match in re.finditer(pattern, cleaned))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return None


def clean_final_label_text(text: str) -> str:
    cleaned = text or ""
    if "<end_of_turn>" in cleaned:
        cleaned = cleaned.split("<end_of_turn>", 1)[0]
    if "</think>" in cleaned:
        tail = cleaned.rsplit("</think>", 1)[1]
        if re.search(r"\b(?:A|B|tie)\b", tail, re.IGNORECASE):
            cleaned = tail
    lowered = cleaned.lower()
    marker_index = max(
        lowered.rfind("[output]"),
        lowered.rfind("output:"),
        lowered.rfind("answer:"),
        lowered.rfind("final answer:"),
        lowered.rfind("result:"),
    )
    if marker_index != -1 and re.search(r"\b(?:A|B|tie)\b", cleaned[marker_index:], re.IGNORECASE):
        cleaned = cleaned[marker_index:]
    return cleaned.strip()


def clean_final_json_text(text: str) -> str:
    cleaned = text or ""
    if "<end_of_turn>" in cleaned:
        cleaned = cleaned.split("<end_of_turn>", 1)[0]
    if "</think>" in cleaned:
        tail = cleaned.rsplit("</think>", 1)[1]
        if "{" in tail or "[" in tail:
            cleaned = tail
    lowered = cleaned.lower()
    marker_index = max(
        lowered.rfind("final json:"),
        lowered.rfind("final answer:"),
        lowered.rfind("[output]"),
        lowered.rfind("output:"),
        lowered.rfind("answer:"),
    )
    if marker_index != -1 and ("{" in cleaned[marker_index:] or "[" in cleaned[marker_index:]):
        cleaned = cleaned[marker_index:]
    return cleaned.strip()


def balanced_json_candidates(text: str) -> list[str]:
    spans: list[tuple[int, int, str]] = []
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        stack: list[str] = []
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
                continue
            if current in "{[":
                stack.append(current)
                continue
            if current in "}]":
                if not stack:
                    break
                opening = stack.pop()
                if (opening, current) not in {("{", "}"), ("[", "]")}:
                    break
                if not stack:
                    spans.append((start, end + 1, text[start : end + 1]))
                    break
    deduped: dict[tuple[int, int], str] = {}
    for start, end, candidate in spans:
        deduped[(start, end)] = candidate
    ordered = sorted(deduped.items(), key=lambda item: (item[0][1], item[0][1] - item[0][0]), reverse=True)
    return [candidate for _, candidate in ordered]


def looks_like_tool_call(payload: Any) -> bool:
    return tr.normalize_tool_call_payload(payload) is not None


def parse_json_like(text: str, *, prefer_tool_call: bool = False) -> Any | None:
    cleaned = clean_final_json_text(text)
    if not cleaned:
        return None
    candidates: list[str] = []
    candidates.extend(
        match.group(1).strip()
        for match in reversed(list(JSON_BLOCK_RE.finditer(cleaned)))
    )
    candidates.extend(balanced_json_candidates(cleaned))
    candidates.append(cleaned)

    first_valid: Any | None = None
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if first_valid is None:
            first_valid = payload
        if prefer_tool_call and looks_like_tool_call(payload):
            return payload
        if not prefer_tool_call:
            return payload
    return first_valid


def normalize_metric_id(metric: Any, valid_metric_ids: list[str]) -> str | None:
    if metric is None:
        return None
    valid_lookup = {metric_id.lower(): metric_id for metric_id in valid_metric_ids}
    cleaned = str(metric).strip().lower()
    if cleaned in valid_lookup:
        return valid_lookup[cleaned]
    slug = slugify(cleaned)
    return valid_lookup.get(slug)


def list_from_topk_payload(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ["top_3", "top3", "top_k", "differences", "metrics", "answer", "results"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def parse_topk_response(
    text: str,
    valid_metric_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]] | None:
    payload = parse_json_like(text)
    entries = list_from_topk_payload(payload)
    if entries is None:
        return None
    parsed: list[dict[str, Any]] = []
    for entry in entries[:top_k]:
        if not isinstance(entry, dict):
            continue
        metric = normalize_metric_id(
            entry.get("metric", entry.get("metric_id", entry.get("name"))),
            valid_metric_ids,
        )
        larger = parse_label(
            str(
                entry.get(
                    "larger",
                    entry.get("larger_image", entry.get("direction", entry.get("winner", ""))),
                )
            ),
            ["A", "B", "tie"],
        )
        parsed.append(
            {
                "metric": metric,
                "larger": larger,
                "raw_metric": entry.get("metric", entry.get("metric_id", entry.get("name"))),
                "raw_larger": entry.get(
                    "larger",
                    entry.get("larger_image", entry.get("direction", entry.get("winner"))),
                ),
            }
        )
    return parsed


def metric_family(metric_id: str) -> str:
    feature = metric_id.split("__", 1)[0]
    if feature in {"area", "area_filled"}:
        return "projected_area"
    if feature in {"area_bbox", "bounding_box_fill", "extent"}:
        return "bounding_box_fill"
    if feature in {"area_convex", "convexity_deficit", "solidity"}:
        return "convexity"
    if feature in {"equivalent_diameter", "feret_diameter_max"}:
        return "particle_diameter"
    if feature in {"major_axis_length", "minor_axis_length"}:
        return "axis_lengths"
    if feature in {"eccentricity", "aspect_ratio", "elongation"}:
        return "elongation"
    if feature in {"circularity", "perimeter_to_diameter_ratio"}:
        return "roundness"
    return feature


def rel_error_pct(prediction: float, truth: float) -> float:
    if truth == 0:
        return 0.0 if prediction == 0 else math.inf
    return abs((prediction - truth) / truth) * 100.0


def smape_pct(prediction: float, truth: float) -> float:
    denom = abs(prediction) + abs(truth)
    if denom == 0:
        return 0.0
    return 200.0 * abs(prediction - truth) / denom


def json_compact(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_expected_options(task: dict[str, Any]) -> list[dict[str, Any]]:
    options = [dict(task.get("expected_arguments") or {})]
    for accepted in task.get("accepted_arguments", []) or []:
        if isinstance(accepted, dict):
            options.append(dict(accepted))
    return options


def tool_field_strict_correct(
    normalized_args: dict[str, Any],
    task: dict[str, Any],
    fields: list[str],
) -> bool:
    expected = dict(task.get("expected_arguments") or {})
    return all(normalized_args.get(field) == expected.get(field) for field in fields)


def tool_field_correct(
    normalized_args: dict[str, Any],
    task: dict[str, Any],
    fields: list[str],
) -> bool:
    for option in tool_expected_options(task):
        if all(normalized_args.get(field) == option.get(field) for field in fields):
            return True
    return False


def tool_required_fields(task_type: str) -> list[str]:
    if task_type == "single_stat":
        return ["image_id", "feature", "statistic"]
    if task_type == "pairwise_compare":
        return ["image_a_id", "image_b_id", "feature", "statistic", "relation"]
    if task_type == "topk_differences":
        return ["image_a_id", "image_b_id", "top_k", "metric_scope"]
    return []


def score_tool_call(
    task: dict[str, Any],
    response: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    score: dict[str, Any] = {
        "tool_json_valid": False,
        "tool_call_valid": False,
        "tool_executable": False,
        "tool_name_correct": False,
        "tool_image_args_correct": False,
        "tool_image_args_strict_correct": False,
        "tool_feature_correct": "",
        "tool_feature_strict_correct": "",
        "tool_statistic_correct": "",
        "tool_statistic_strict_correct": "",
        "tool_relation_correct": "",
        "tool_relation_strict_correct": "",
        "tool_top_k_correct": "",
        "tool_top_k_strict_correct": "",
        "tool_metric_scope_correct": "",
        "tool_metric_scope_strict_correct": "",
        "tool_routing_correct": False,
        "tool_routing_strict_correct": False,
        "tool_result_correct": False,
        "tool_predicted_tool": "",
        "tool_predicted_arguments": "",
        "tool_expected_tool": task.get("expected_tool", ""),
        "tool_expected_arguments": json_compact(task.get("expected_arguments", {})),
        "tool_accepted_arguments": json_compact(task.get("accepted_arguments", [])),
        "tool_execution_result": "",
        "tool_error": "",
    }

    payload = parse_json_like(response, prefer_tool_call=True)
    if payload is None:
        score["tool_error"] = "invalid_json"
        return score
    score["tool_json_valid"] = True

    call = tr.normalize_tool_call_payload(payload)
    if call is None:
        score["tool_error"] = "invalid_tool_call"
        return score
    score["tool_call_valid"] = True
    tool = str(call["tool"])
    score["tool_predicted_tool"] = tool
    score["tool_name_correct"] = tool == task.get("expected_tool")

    normalized_args, argument_error = tr.normalize_tool_arguments(
        tool,
        dict(call.get("arguments") or {}),
        config,
    )
    score["tool_predicted_arguments"] = json_compact(normalized_args)
    if argument_error:
        score["tool_error"] = argument_error
        return score

    task_type = str(task.get("tool_task_type"))
    if task_type == "single_stat":
        score["tool_image_args_correct"] = tool_field_correct(normalized_args, task, ["image_id"])
        score["tool_image_args_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["image_id"],
        )
        score["tool_feature_correct"] = tool_field_correct(normalized_args, task, ["feature"])
        score["tool_feature_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["feature"],
        )
        score["tool_statistic_correct"] = tool_field_correct(normalized_args, task, ["statistic"])
        score["tool_statistic_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["statistic"],
        )
    elif task_type == "pairwise_compare":
        score["tool_image_args_correct"] = tool_field_correct(
            normalized_args,
            task,
            ["image_a_id", "image_b_id"],
        )
        score["tool_image_args_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["image_a_id", "image_b_id"],
        )
        score["tool_feature_correct"] = tool_field_correct(normalized_args, task, ["feature"])
        score["tool_feature_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["feature"],
        )
        score["tool_statistic_correct"] = tool_field_correct(normalized_args, task, ["statistic"])
        score["tool_statistic_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["statistic"],
        )
        score["tool_relation_correct"] = tool_field_correct(normalized_args, task, ["relation"])
        score["tool_relation_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["relation"],
        )
    elif task_type == "topk_differences":
        score["tool_image_args_correct"] = tool_field_correct(
            normalized_args,
            task,
            ["image_a_id", "image_b_id"],
        )
        score["tool_image_args_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["image_a_id", "image_b_id"],
        )
        score["tool_top_k_correct"] = tool_field_correct(normalized_args, task, ["top_k"])
        score["tool_top_k_strict_correct"] = tool_field_strict_correct(normalized_args, task, ["top_k"])
        score["tool_metric_scope_correct"] = tool_field_correct(normalized_args, task, ["metric_scope"])
        score["tool_metric_scope_strict_correct"] = tool_field_strict_correct(
            normalized_args,
            task,
            ["metric_scope"],
        )

    required_fields = tool_required_fields(task_type)
    score["tool_routing_correct"] = bool(
        score["tool_name_correct"]
        and required_fields
        and tool_field_correct(normalized_args, task, required_fields)
    )
    score["tool_routing_strict_correct"] = bool(
        score["tool_name_correct"]
        and required_fields
        and tool_field_strict_correct(normalized_args, task, required_fields)
    )

    try:
        result = tr.execute_normalized_tool_call(
            config,
            tool,
            normalized_args,
            scale_image_ids=[int(item) for item in task.get("scale_image_ids", [])] or None,
        )
    except Exception as exc:  # noqa: BLE001 - persisted scoring diagnostic.
        score["tool_error"] = f"execution_failed:{exc}"
        return score

    score["tool_executable"] = True
    score["tool_execution_result"] = json_compact(result)
    score["tool_result_correct"] = bool(tr.result_matches(result, task.get("ground_truth"), task_type))
    return score


def discover_model_labels(raw_root: Path, run_id: str) -> list[str]:
    labels: list[str] = []
    if not raw_root.exists():
        return labels
    for model_dir in sorted(raw_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if (model_dir / slugify(run_id)).exists():
            labels.append(model_dir.name)
    return labels


def discover_model_labels_for_runs(raw_root: Path, run_ids: list[str]) -> list[str]:
    labels: list[str] = []
    if not raw_root.exists():
        return labels
    run_slugs = {slugify(run_id) for run_id in run_ids}
    for model_dir in sorted(raw_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if any((model_dir / run_slug).exists() for run_slug in run_slugs):
            labels.append(model_dir.name)
    return labels


def read_output(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"status": "error", "error": "unreadable output json"}


def output_status(path: Path) -> str:
    output = read_output(path)
    if output is None:
        return "missing"
    return str(output.get("status", "unknown"))


def count_matching_outputs(
    model_run_dir: Path,
    manifest_tasks: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = {"success": 0, "available": 0}
    if not model_run_dir.exists():
        return counts
    for task_id in manifest_tasks:
        status = output_status(model_run_dir / f"{slugify(task_id)}.json")
        if status != "missing":
            counts["available"] += 1
        if status == "success":
            counts["success"] += 1
    return counts


def choose_run_id(
    raw_root: Path,
    model_label: str,
    manifest_tasks: dict[str, dict[str, Any]],
    run_ids: list[str],
) -> tuple[str, dict[str, dict[str, int]]]:
    model_root = raw_root / slugify(model_label)
    counts_by_run: dict[str, dict[str, int]] = {}
    for run_id in run_ids:
        counts_by_run[run_id] = count_matching_outputs(
            model_root / slugify(run_id),
            manifest_tasks,
        )
    best_run = max(
        run_ids,
        key=lambda run_id: (
            counts_by_run[run_id]["success"],
            counts_by_run[run_id]["available"],
            -run_ids.index(run_id),
        ),
    )
    return best_run, counts_by_run


def score_task(
    task: dict[str, Any],
    output: dict[str, Any] | None,
    model_label: str,
    source_run_id: str,
    thresholds: list[float],
    numeric_parser: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_label": model_label,
        "source_run_id": source_run_id,
        "numeric_parser": numeric_parser,
        "task_id": task["task_id"],
        "suite": task.get("suite"),
        "answer_type": task.get("answer_type"),
        "encoding": task.get("encoding"),
        "feature": task.get("feature"),
        "statistic": task.get("statistic"),
        "ground_truth": task.get("ground_truth"),
        "case_study_stage": task.get("case_study_stage", ""),
        "image_id": task.get("image_id", ""),
        "image_name": task.get("image_name", ""),
        "image_a_id": task.get("image_a_id", ""),
        "image_b_id": task.get("image_b_id", ""),
        "image_a_name": task.get("image_a_name", ""),
        "image_b_name": task.get("image_b_name", ""),
        "distance_stratum": task.get("distance_stratum", ""),
        "pair_distance": task.get("pair_distance", ""),
        "repeat_group_id": task.get("repeat_group_id", ""),
        "repeat_index": task.get("repeat_index", ""),
        "original_task_id": task.get("original_task_id", ""),
        "tool_task_type": task.get("tool_task_type", ""),
        "tool_intent_key": task.get("tool_intent_key", ""),
        "paraphrase_id": task.get("paraphrase_id", ""),
        "paraphrase_family": task.get("paraphrase_family", ""),
        "status": "missing",
        "response": "",
        "parse_success": False,
        "correct": False,
        "prediction": "",
        "abs_error": "",
        "rel_error_pct": "",
        "smape_pct": "",
        "topk_precision_at_3": "",
        "topk_direction_accuracy_at_3": "",
        "topk_exact_set_match": "",
        "topk_family_precision_at_3": "",
        "topk_valid_metric_rate": "",
        "topk_predicted_metrics": "",
        "topk_ground_truth_metrics": "",
        "tool_json_valid": "",
        "tool_call_valid": "",
        "tool_executable": "",
        "tool_name_correct": "",
        "tool_image_args_correct": "",
        "tool_image_args_strict_correct": "",
        "tool_feature_correct": "",
        "tool_feature_strict_correct": "",
        "tool_statistic_correct": "",
        "tool_statistic_strict_correct": "",
        "tool_relation_correct": "",
        "tool_relation_strict_correct": "",
        "tool_top_k_correct": "",
        "tool_top_k_strict_correct": "",
        "tool_metric_scope_correct": "",
        "tool_metric_scope_strict_correct": "",
        "tool_routing_correct": "",
        "tool_routing_strict_correct": "",
        "tool_result_correct": "",
        "tool_predicted_tool": "",
        "tool_predicted_arguments": "",
        "tool_expected_tool": task.get("expected_tool", ""),
        "tool_expected_arguments": json_compact(task.get("expected_arguments", {}))
        if task.get("expected_arguments") is not None
        else "",
        "tool_accepted_arguments": json_compact(task.get("accepted_arguments", [])),
        "tool_execution_result": "",
        "tool_error": "",
        "elapsed_seconds": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "reasoning_tokens": "",
    }
    for threshold in thresholds:
        row[f"correct_within_{threshold:g}pct"] = False

    if output is None:
        return row

    row["status"] = output.get("status", "unknown")
    row["response"] = output.get("response", "")
    runtime = output.get("runtime") or {}
    usage = output.get("usage") or {}
    row["elapsed_seconds"] = runtime.get("elapsed_seconds", "")
    row["input_tokens"] = usage.get("input_tokens", "")
    row["output_tokens"] = usage.get("output_tokens", "")
    row["total_tokens"] = usage.get("total_tokens", "")
    row["reasoning_tokens"] = usage.get("reasoning_tokens", "")

    if row["status"] != "success":
        return row

    if task.get("answer_type") == "numeric":
        prediction = parse_number_with_mode(row["response"], numeric_parser)
        if prediction is None:
            return row
        truth = float(task["ground_truth"])
        row["parse_success"] = True
        row["prediction"] = prediction
        row["abs_error"] = abs(prediction - truth)
        row["rel_error_pct"] = rel_error_pct(prediction, truth)
        row["smape_pct"] = smape_pct(prediction, truth)
        for threshold in thresholds:
            is_correct = row["rel_error_pct"] <= threshold
            row[f"correct_within_{threshold:g}pct"] = is_correct
        row["correct"] = row["correct_within_1pct"] if "correct_within_1pct" in row else False
        return row

    if task.get("answer_type") == "label":
        valid_labels = task.get("valid_labels", ["A", "B", "tie"])
        prediction = parse_label(row["response"], valid_labels)
        if prediction is None:
            return row
        row["parse_success"] = True
        row["prediction"] = prediction
        row["correct"] = prediction == task["ground_truth"]
        for threshold in thresholds:
            row[f"correct_within_{threshold:g}pct"] = row["correct"]
        return row

    if task.get("answer_type") == "topk_metrics":
        valid_metric_ids = [str(item) for item in task.get("valid_metric_ids", [])]
        top_k = int(task.get("top_k", 3))
        truth_items = task.get("ground_truth", [])
        truth_metrics = [str(item.get("metric")) for item in truth_items if isinstance(item, dict)]
        truth_families = [
            str(item.get("family", metric_family(str(item.get("metric")))))
            for item in truth_items
            if isinstance(item, dict)
        ]
        truth_directions = {
            str(item.get("metric")): item.get("larger")
            for item in truth_items
            if isinstance(item, dict)
        }
        row["topk_ground_truth_metrics"] = ";".join(truth_metrics)

        parsed = parse_topk_response(row["response"], valid_metric_ids, top_k)
        if parsed is None:
            return row

        row["parse_success"] = True
        parsed = parsed[:top_k]
        valid_predictions = [
            item for item in parsed if item.get("metric") in valid_metric_ids
        ]
        predicted_metrics = [
            str(item["metric"]) for item in valid_predictions if item.get("metric") is not None
        ]
        predicted_families = [metric_family(metric) for metric in predicted_metrics[:top_k]]
        row["prediction"] = ";".join(predicted_metrics)
        row["topk_predicted_metrics"] = ";".join(predicted_metrics)
        row["topk_valid_metric_rate"] = (
            len(valid_predictions) / len(parsed) if parsed else 0.0
        )

        truth_set = set(truth_metrics[:top_k])
        predicted_set = set(predicted_metrics[:top_k])
        overlap = predicted_set & truth_set
        direction_matches = 0
        for item in valid_predictions[:top_k]:
            metric = str(item.get("metric"))
            if metric in truth_directions and item.get("larger") == truth_directions[metric]:
                direction_matches += 1

        precision = len(overlap) / top_k if top_k else 0.0
        family_precision = (
            len(set(predicted_families) & set(truth_families[:top_k])) / top_k if top_k else 0.0
        )
        direction_accuracy = direction_matches / top_k if top_k else 0.0
        exact_set_match = predicted_set == truth_set and len(predicted_set) == top_k
        row["topk_precision_at_3"] = precision
        row["topk_family_precision_at_3"] = family_precision
        row["topk_direction_accuracy_at_3"] = direction_accuracy
        row["topk_exact_set_match"] = exact_set_match
        row["correct"] = exact_set_match and direction_accuracy == 1.0
        for threshold in thresholds:
            row[f"correct_within_{threshold:g}pct"] = row["correct"]
        return row

    if task.get("answer_type") == "tool_call":
        tool_score = score_tool_call(task, row["response"], config)
        row.update(tool_score)
        row["parse_success"] = bool(tool_score["tool_call_valid"])
        row["prediction"] = tool_score["tool_predicted_arguments"]
        row["correct"] = bool(
            tool_score["tool_routing_correct"] and tool_score["tool_result_correct"]
        )
        for threshold in thresholds:
            row[f"correct_within_{threshold:g}pct"] = row["correct"]
        return row

    return row


def bootstrap_ci(values: list[bool], iterations: int, seed: int) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = []
    for _ in range(iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(sample)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"low": float(low), "high": float(high)}


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    thresholds = config["evaluation"]["threshold_sweep_pct"]
    iterations = int(config["evaluation"].get("bootstrap_iterations", 1000))
    seed = int(config["evaluation"].get("bootstrap_seed", 20260805))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_label"])].append(row)

    for model_label, model_rows in grouped.items():
        success_rows = [row for row in model_rows if row["status"] == "success"]
        parsed_rows = [row for row in success_rows if row["parse_success"]]
        correct_values = [bool(row["correct"]) for row in parsed_rows]
        usage_totals = Counter()
        elapsed_values = []
        rel_errors = []
        smape_values = []
        topk_precision_values = []
        topk_family_precision_values = []
        topk_direction_values = []
        topk_valid_metric_values = []
        topk_exact_values = []
        tool_metric_values: dict[str, list[float]] = defaultdict(list)
        for row in success_rows:
            if row.get("answer_type") != "tool_call":
                continue
            for field in [
                "tool_json_valid",
                "tool_call_valid",
                "tool_executable",
                "tool_name_correct",
                "tool_image_args_correct",
                "tool_image_args_strict_correct",
                "tool_routing_correct",
                "tool_routing_strict_correct",
                "tool_result_correct",
            ]:
                if isinstance(row.get(field), bool):
                    tool_metric_values[field].append(float(row[field]))
            for field in [
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
            ]:
                if isinstance(row.get(field), bool):
                    tool_metric_values[field].append(float(row[field]))
        for row in parsed_rows:
            for field in ["input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"]:
                value = row.get(field)
                if isinstance(value, (int, float)):
                    usage_totals[field] += float(value)
            if isinstance(row.get("elapsed_seconds"), (int, float)):
                elapsed_values.append(float(row["elapsed_seconds"]))
            if isinstance(row.get("rel_error_pct"), (int, float)) and math.isfinite(row["rel_error_pct"]):
                rel_errors.append(float(row["rel_error_pct"]))
            if isinstance(row.get("smape_pct"), (int, float)) and math.isfinite(row["smape_pct"]):
                smape_values.append(float(row["smape_pct"]))
            if isinstance(row.get("topk_precision_at_3"), (int, float)):
                topk_precision_values.append(float(row["topk_precision_at_3"]))
            if isinstance(row.get("topk_family_precision_at_3"), (int, float)):
                topk_family_precision_values.append(float(row["topk_family_precision_at_3"]))
            if isinstance(row.get("topk_direction_accuracy_at_3"), (int, float)):
                topk_direction_values.append(float(row["topk_direction_accuracy_at_3"]))
            if isinstance(row.get("topk_valid_metric_rate"), (int, float)):
                topk_valid_metric_values.append(float(row["topk_valid_metric_rate"]))
            if isinstance(row.get("topk_exact_set_match"), bool):
                topk_exact_values.append(float(row["topk_exact_set_match"]))

        threshold_rates = {}
        for threshold in thresholds:
            key = f"correct_within_{threshold:g}pct"
            values = [bool(row[key]) for row in parsed_rows if key in row]
            threshold_rates[key] = float(np.mean(values)) if values else None

        by_model[model_label] = {
            "total_tasks": len(model_rows),
            "successful_outputs": len(success_rows),
            "parsed_outputs": len(parsed_rows),
            "parse_failure_rate": 1.0 - (len(parsed_rows) / len(success_rows)) if success_rows else None,
            "accuracy": float(np.mean(correct_values)) if correct_values else None,
            "accuracy_ci95": bootstrap_ci(correct_values, iterations, seed),
            "threshold_rates": threshold_rates,
            "mean_rel_error_pct": float(np.mean(rel_errors)) if rel_errors else None,
            "mean_smape_pct": float(np.mean(smape_values)) if smape_values else None,
            "mean_topk_precision_at_3": (
                float(np.mean(topk_precision_values)) if topk_precision_values else None
            ),
            "mean_topk_direction_accuracy_at_3": (
                float(np.mean(topk_direction_values)) if topk_direction_values else None
            ),
            "mean_topk_family_precision_at_3": (
                float(np.mean(topk_family_precision_values))
                if topk_family_precision_values
                else None
            ),
            "mean_topk_valid_metric_rate": (
                float(np.mean(topk_valid_metric_values)) if topk_valid_metric_values else None
            ),
            "topk_exact_set_match_rate": (
                float(np.mean(topk_exact_values)) if topk_exact_values else None
            ),
            "tool_json_valid_rate": (
                float(np.mean(tool_metric_values["tool_json_valid"]))
                if tool_metric_values["tool_json_valid"]
                else None
            ),
            "tool_call_valid_rate": (
                float(np.mean(tool_metric_values["tool_call_valid"]))
                if tool_metric_values["tool_call_valid"]
                else None
            ),
            "tool_executable_rate": (
                float(np.mean(tool_metric_values["tool_executable"]))
                if tool_metric_values["tool_executable"]
                else None
            ),
            "tool_routing_accuracy": (
                float(np.mean(tool_metric_values["tool_routing_correct"]))
                if tool_metric_values["tool_routing_correct"]
                else None
            ),
            "tool_routing_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_routing_strict_correct"]))
                if tool_metric_values["tool_routing_strict_correct"]
                else None
            ),
            "tool_result_accuracy": (
                float(np.mean(tool_metric_values["tool_result_correct"]))
                if tool_metric_values["tool_result_correct"]
                else None
            ),
            "tool_name_accuracy": (
                float(np.mean(tool_metric_values["tool_name_correct"]))
                if tool_metric_values["tool_name_correct"]
                else None
            ),
            "tool_image_arg_accuracy": (
                float(np.mean(tool_metric_values["tool_image_args_correct"]))
                if tool_metric_values["tool_image_args_correct"]
                else None
            ),
            "tool_image_arg_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_image_args_strict_correct"]))
                if tool_metric_values["tool_image_args_strict_correct"]
                else None
            ),
            "tool_feature_accuracy": (
                float(np.mean(tool_metric_values["tool_feature_correct"]))
                if tool_metric_values["tool_feature_correct"]
                else None
            ),
            "tool_feature_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_feature_strict_correct"]))
                if tool_metric_values["tool_feature_strict_correct"]
                else None
            ),
            "tool_statistic_accuracy": (
                float(np.mean(tool_metric_values["tool_statistic_correct"]))
                if tool_metric_values["tool_statistic_correct"]
                else None
            ),
            "tool_statistic_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_statistic_strict_correct"]))
                if tool_metric_values["tool_statistic_strict_correct"]
                else None
            ),
            "tool_relation_accuracy": (
                float(np.mean(tool_metric_values["tool_relation_correct"]))
                if tool_metric_values["tool_relation_correct"]
                else None
            ),
            "tool_relation_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_relation_strict_correct"]))
                if tool_metric_values["tool_relation_strict_correct"]
                else None
            ),
            "tool_top_k_accuracy": (
                float(np.mean(tool_metric_values["tool_top_k_correct"]))
                if tool_metric_values["tool_top_k_correct"]
                else None
            ),
            "tool_top_k_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_top_k_strict_correct"]))
                if tool_metric_values["tool_top_k_strict_correct"]
                else None
            ),
            "tool_metric_scope_accuracy": (
                float(np.mean(tool_metric_values["tool_metric_scope_correct"]))
                if tool_metric_values["tool_metric_scope_correct"]
                else None
            ),
            "tool_metric_scope_strict_accuracy": (
                float(np.mean(tool_metric_values["tool_metric_scope_strict_correct"]))
                if tool_metric_values["tool_metric_scope_strict_correct"]
                else None
            ),
            "mean_elapsed_seconds": float(np.mean(elapsed_values)) if elapsed_values else None,
            "usage_totals": dict(usage_totals),
        }
    return {"by_model": by_model}


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


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiments_config)
    if not args.manifest:
        image_limit = int(config["image_policy"]["default_limit"])
        args.manifest = f"{config['paths']['manifests_dir']}/tasks_image{image_limit}.jsonl"
    raw_root = REPO_ROOT / (args.raw_root or config["paths"]["raw_results_dir"])
    evaluation_dir = REPO_ROOT / config["paths"]["evaluation_dir"]
    manifest_tasks = {task["task_id"]: task for task in iter_jsonl(args.manifest)}
    thresholds = [float(item) for item in config["evaluation"]["threshold_sweep_pct"]]

    auto_run = args.run_id == "auto"
    run_id_candidates = args.run_id_candidate or DEFAULT_RUN_ID_CANDIDATES
    model_labels = args.model_label or (
        discover_model_labels_for_runs(raw_root, run_id_candidates)
        if auto_run
        else discover_model_labels(raw_root, args.run_id)
    )
    if not model_labels:
        raise SystemExit("No model outputs found. Pass --model-label or run inference first.")

    rows: list[dict[str, Any]] = []
    run_selection: dict[str, Any] = {}
    explicit_model_labels = args.model_label is not None
    for model_label in model_labels:
        if auto_run:
            selected_run_id, counts_by_run = choose_run_id(
                raw_root,
                model_label,
                manifest_tasks,
                run_id_candidates,
            )
            run_selection[model_label] = {
                "selected_run_id": selected_run_id,
                "counts_by_run": counts_by_run,
            }
            count = counts_by_run[selected_run_id]
            if not explicit_model_labels and count["available"] == 0:
                print(f"{model_label}: skipped; no matching outputs in auto candidates")
                continue
            print(
                f"{model_label}: selected {selected_run_id} "
                f"({count['success']:,} success, {count['available']:,} available)"
            )
        else:
            selected_run_id = args.run_id
        model_dir = raw_root / slugify(model_label) / slugify(selected_run_id)
        for task_id, task in manifest_tasks.items():
            output = read_output(model_dir / f"{slugify(task_id)}.json")
            rows.append(
                score_task(
                    task,
                    output,
                    model_label,
                    selected_run_id,
                    thresholds,
                    args.numeric_parser,
                    config,
                )
            )

    prefix = args.output_prefix or ("auto_best" if auto_run else f"{slugify(args.run_id)}")
    csv_path = evaluation_dir / f"{prefix}_scores.csv"
    summary_path = evaluation_dir / f"{prefix}_summary.json"
    write_csv(csv_path, rows)
    summary = summarize(rows, config)
    if auto_run:
        summary["run_selection"] = run_selection
    write_json_atomic(summary_path, summary)
    print(f"Wrote scores: {csv_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
