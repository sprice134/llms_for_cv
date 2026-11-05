#!/usr/bin/env python3
"""Build task manifests for the AI4Mat characterization-interface benchmark."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from ai4mat_common import (
    REPO_ROOT,
    as_float,
    load_yaml,
    read_json,
    selected_image_ids,
    slugify,
    write_jsonl,
)
import tool_routing as tr


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument(
        "--suite",
        choices=["core", "interface", "case_study", "tool_routing", "all"],
        default="all",
        help="Task suite to build. Default: all",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Override configured image count, e.g. 15 or 20.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional task-record limit for smoke manifests.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to results/manifests/tasks_image<N>.jsonl.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_processed_row(processed_path: Path, feature: str) -> dict[str, str]:
    with processed_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Metric") == feature:
                return row
    raise KeyError(f"Feature {feature!r} not found in {processed_path}")


def read_raw_feature(raw_path: Path, feature: str) -> list[float]:
    values: list[float] = []
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if feature not in (reader.fieldnames or []):
            raise KeyError(f"Feature {feature!r} not found in {raw_path}")
        for row in reader:
            value = row.get(feature)
            if value in (None, ""):
                continue
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    if not values:
        raise ValueError(f"No finite values for {feature!r} in {raw_path}")
    return values


def read_raw_numeric_rows(raw_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            numeric_row: dict[str, float] = {}
            for key, value in row.items():
                if key == "object_id" or value in (None, ""):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if math.isfinite(number):
                    numeric_row[key] = number
            rows.append(numeric_row)
    if not rows:
        raise ValueError(f"No numeric rows found in {raw_path}")
    return rows


def numeric_stat(values: list[float], statistic: str) -> float:
    arr = np.asarray(values, dtype=float)
    if statistic == "mean":
        return float(np.mean(arr))
    if statistic == "median" or statistic == "percentile_50":
        return float(np.quantile(arr, 0.50))
    if statistic == "range":
        return float(np.max(arr) - np.min(arr))
    if statistic == "min":
        return float(np.min(arr))
    if statistic == "max":
        return float(np.max(arr))
    if statistic == "iqr":
        return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    if statistic == "q1":
        return float(np.quantile(arr, 0.25))
    if statistic == "q3":
        return float(np.quantile(arr, 0.75))
    if statistic == "percentile_10":
        return float(np.quantile(arr, 0.10))
    if statistic == "percentile_90":
        return float(np.quantile(arr, 0.90))
    if statistic == "d90_minus_d10_over_d50":
        d10 = np.quantile(arr, 0.10)
        d50 = np.quantile(arr, 0.50)
        d90 = np.quantile(arr, 0.90)
        if d50 == 0:
            raise ValueError("Cannot calculate span with D50=0")
        return float((d90 - d10) / d50)
    if statistic == "std":
        return float(np.std(arr))
    if statistic == "coefficient_of_variation":
        mean = float(np.mean(arr))
        if mean == 0:
            raise ValueError("Cannot calculate coefficient of variation with mean=0")
        return float(np.std(arr) / abs(mean))
    raise KeyError(f"Unsupported statistic: {statistic}")


def source_path(morphology_root: Path, image_name: str, suffix: str) -> Path:
    return morphology_root / image_name / f"{image_name}{suffix}"


def core_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    paths = config["paths"]
    image_map = {int(k): v for k, v in read_json(paths["images_map"]).items()}
    morphology_root = REPO_ROOT / paths["morphology_root"]
    core = config["core_benchmark"]
    tasks: list[dict[str, Any]] = []

    for image_id, feature, stat, encoding in itertools.product(
        image_ids,
        core["features"],
        core["statistics"],
        core["encodings"],
    ):
        image_name = image_map[image_id]
        stat_key = stat["key"]
        prompt_stat = stat.get("prompt_label", stat_key)
        processed_path = source_path(morphology_root, image_name, "_processed.csv")
        processed_row = load_processed_row(processed_path, feature)
        ground_truth = as_float(processed_row[stat["processed_column"]])

        data_path = source_path(morphology_root, image_name, encoding["suffix"])
        data_text = read_text(data_path)
        prompt = encoding["prompt_template"].format(
            statistic=prompt_stat,
            feature=feature,
            data=data_text,
        )
        task_id = "__".join(
            [
                "core",
                f"img{image_id:02d}",
                slugify(encoding["key"]),
                slugify(feature),
                slugify(stat_key),
            ]
        )
        tasks.append(
            {
                "task_id": task_id,
                "suite": "core",
                "answer_type": "numeric",
                "image_id": image_id,
                "image_name": image_name,
                "encoding": encoding["key"],
                "legacy_encoding_label": encoding["legacy_label"],
                "data_kind": encoding["data_kind"],
                "serialization": encoding["serialization"],
                "feature": feature,
                "statistic": stat_key,
                "prompt_statistic": prompt_stat,
                "ground_truth": ground_truth,
                "units": infer_units(feature),
                "prompt": prompt,
                "source_paths": [rel(data_path), rel(processed_path)],
            }
        )
    return tasks


def infer_units(feature: str) -> str:
    if feature in {"eccentricity", "solidity", "extent", "circularity"}:
        return "unitless"
    if feature == "orientation":
        return "radians"
    if "area" in feature:
        return "pixels_squared"
    return "pixels"


def values_payload(image_name: str, feature: str, values: list[float]) -> str:
    rounded = [round(value, 6) for value in values]
    return json.dumps(
        {"image": image_name, "feature": feature, "values": rounded},
        separators=(",", ":"),
    )


def routed_fact_prompt(label: str, facts: dict[str, Any]) -> str:
    data = json.dumps(facts, indent=2, sort_keys=True)
    return (
        "Here are deterministic SEM powder characterization facts as JSON.\n"
        f"{data}\n"
        f"What is the {label}? Give only the requested number or label."
    )


def interface_numeric_tasks(
    config: dict[str, Any],
    image_ids: list[int],
    *,
    suite_name: str = "interface_numeric",
    task_prefix: str = "interface_numeric",
    case_study_stage: int | None = None,
) -> list[dict[str, Any]]:
    paths = config["paths"]
    image_map = {int(k): v for k, v in read_json(paths["images_map"]).items()}
    morphology_root = REPO_ROOT / paths["morphology_root"]
    suite = config["interface_suite"]
    tasks: list[dict[str, Any]] = []

    for image_id, task_def, mode in itertools.product(
        image_ids,
        suite["numeric_tasks"],
        suite["modes"],
    ):
        image_name = image_map[image_id]
        feature = task_def["feature"]
        statistic = task_def["statistic"]
        raw_path = source_path(morphology_root, image_name, "_raw.csv")
        raw_text_path = source_path(morphology_root, image_name, "_raw.txt")
        values = read_raw_feature(raw_path, feature)
        truth = numeric_stat(values, statistic)

        if mode["key"] == "routed_json":
            facts = {
                "image": image_name,
                "feature": feature,
                "statistic": statistic,
                "value": truth,
                "units": task_def.get("units", infer_units(feature)),
            }
            prompt = routed_fact_prompt(task_def["prompt_label"], facts)
            source_paths = [rel(raw_path)]
        elif mode["key"] == "raw_text":
            data = values_payload(image_name, feature, values)
            prompt = (
                "Here is per-particle SEM powder morphology data as JSON.\n"
                f"{data}\n"
                f"What is the {task_def['prompt_label']}? Give only the number."
            )
            source_paths = [rel(raw_text_path), rel(raw_path)]
        else:
            raise KeyError(f"Unsupported interface mode: {mode['key']}")

        task_id = "__".join(
            [
                task_prefix,
                f"img{image_id:02d}",
                slugify(mode["key"]),
                slugify(task_def["key"]),
            ]
        )
        record = {
            "task_id": task_id,
            "suite": suite_name,
            "answer_type": "numeric",
            "image_id": image_id,
            "image_name": image_name,
            "encoding": mode["key"],
            "feature": feature,
            "statistic": statistic,
            "prompt_statistic": task_def["prompt_label"],
            "ground_truth": truth,
            "units": task_def.get("units", infer_units(feature)),
            "prompt": prompt,
            "source_paths": source_paths,
        }
        if case_study_stage is not None:
            record["case_study_stage"] = case_study_stage
        tasks.append(record)
    return tasks


def interface_pairwise_tasks(
    config: dict[str, Any],
    image_ids: list[int],
    *,
    suite_name: str = "interface_pairwise",
    task_prefix: str = "interface_pairwise",
    case_study_stage: int | None = None,
) -> list[dict[str, Any]]:
    paths = config["paths"]
    image_map = {int(k): v for k, v in read_json(paths["images_map"]).items()}
    morphology_root = REPO_ROOT / paths["morphology_root"]
    suite = config["interface_suite"]
    pairwise = suite["pairwise_tasks"]
    if not pairwise.get("enabled", True):
        return []

    image_pairs = list(itertools.combinations(image_ids, 2))[: int(pairwise["max_pairs"])]
    tasks: list[dict[str, Any]] = []
    for pair_index, ((image_a, image_b), comparison, mode) in enumerate(
        itertools.product(image_pairs, pairwise["comparisons"], suite["modes"]),
        start=1,
    ):
        name_a = image_map[image_a]
        name_b = image_map[image_b]
        feature = comparison["feature"]
        stat = comparison["statistic"]
        raw_a = source_path(morphology_root, name_a, "_raw.csv")
        raw_b = source_path(morphology_root, name_b, "_raw.csv")
        values_a = read_raw_feature(raw_a, feature)
        values_b = read_raw_feature(raw_b, feature)
        metric_a = numeric_stat(values_a, stat)
        metric_b = numeric_stat(values_b, stat)
        if math.isclose(metric_a, metric_b, rel_tol=1e-12, abs_tol=1e-12):
            label = "tie"
        else:
            label = "A" if metric_a > metric_b else "B"

        if mode["key"] == "routed_json":
            facts = {
                "A": {"image": name_a, "feature": feature, "statistic": stat, "value": metric_a},
                "B": {"image": name_b, "feature": feature, "statistic": stat, "value": metric_b},
                "valid_answers": ["A", "B", "tie"],
            }
            prompt = routed_fact_prompt(comparison["prompt_label"], facts)
            source_paths = [rel(raw_a), rel(raw_b)]
        elif mode["key"] == "raw_text":
            payload = {
                "A": json.loads(values_payload(name_a, feature, values_a)),
                "B": json.loads(values_payload(name_b, feature, values_b)),
                "valid_answers": ["A", "B", "tie"],
            }
            prompt = (
                "Here are per-particle SEM powder morphology values for two images as JSON.\n"
                f"{json.dumps(payload, separators=(',', ':'))}\n"
                f"Which image has the {comparison['prompt_label']}? "
                "Answer only A, B, or tie."
            )
            source_paths = [rel(raw_a), rel(raw_b)]
        else:
            raise KeyError(f"Unsupported interface mode: {mode['key']}")

        task_id = "__".join(
            [
                task_prefix,
                f"p{pair_index:03d}",
                slugify(mode["key"]),
                slugify(comparison["key"]),
            ]
        )
        record = {
            "task_id": task_id,
            "suite": suite_name,
            "answer_type": "label",
            "image_a_id": image_a,
            "image_b_id": image_b,
            "image_a_name": name_a,
            "image_b_name": name_b,
            "encoding": mode["key"],
            "feature": feature,
            "statistic": stat,
            "prompt_statistic": comparison["prompt_label"],
            "ground_truth": label,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "valid_labels": ["A", "B", "tie"],
            "prompt": prompt,
            "source_paths": source_paths,
        }
        if case_study_stage is not None:
            record["case_study_stage"] = case_study_stage
        tasks.append(record)
    return tasks


STAGE2_DIRECT_FEATURES = [
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "major_axis_length",
    "minor_axis_length",
    "equivalent_diameter",
    "feret_diameter_max",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
]

STAGE2_DERIVED_FEATURES = [
    "aspect_ratio",
    "elongation",
    "convexity_deficit",
    "bounding_box_fill",
    "perimeter_to_diameter_ratio",
]

STAGE2_STATISTICS = [
    "median",
    "iqr",
    "coefficient_of_variation",
    "percentile_10",
    "percentile_90",
    "d90_minus_d10_over_d50",
]

STAT_ID_MAP = {
    "median": "median",
    "iqr": "iqr",
    "coefficient_of_variation": "cv",
    "percentile_10": "d10",
    "percentile_90": "d90",
    "d90_minus_d10_over_d50": "span_d90_d10_d50",
}

STAT_LABEL_MAP = {
    "median": "median",
    "iqr": "interquartile range",
    "coefficient_of_variation": "coefficient of variation",
    "percentile_10": "D10",
    "percentile_90": "D90",
    "d90_minus_d10_over_d50": "(D90-D10)/D50 span",
}


def stage2_config(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("case_study", {}).get("stage2_topk", {})
    return {
        "pair_count": int(configured.get("pair_count", 30)),
        "pairs_per_stratum": int(configured.get("pairs_per_stratum", 10)),
        "top_k": int(configured.get("top_k", 3)),
        "random_seed": int(configured.get("random_seed", config["project"]["random_seed"])),
        "direct_features": configured.get("direct_features", STAGE2_DIRECT_FEATURES),
        "derived_features": configured.get("derived_features", STAGE2_DERIVED_FEATURES),
        "statistics": configured.get("statistics", STAGE2_STATISTICS),
    }


def derived_value(feature: str, row: dict[str, float]) -> float | None:
    area = row.get("area")
    area_bbox = row.get("area_bbox")
    area_convex = row.get("area_convex")
    major = row.get("major_axis_length")
    minor = row.get("minor_axis_length")
    perimeter = row.get("perimeter")
    equivalent_diameter = row.get("equivalent_diameter")

    if feature == "aspect_ratio" and major is not None and minor not in (None, 0):
        return major / minor
    if feature == "elongation" and major not in (None, 0) and minor is not None:
        return 1.0 - (minor / major)
    if feature == "convexity_deficit" and area_convex not in (None, 0) and area is not None:
        return (area_convex - area) / area_convex
    if feature == "bounding_box_fill" and area_bbox not in (None, 0) and area is not None:
        return area / area_bbox
    if (
        feature == "perimeter_to_diameter_ratio"
        and perimeter is not None
        and equivalent_diameter not in (None, 0)
    ):
        return perimeter / equivalent_diameter
    return None


def stage2_feature_values(
    rows: list[dict[str, float]],
    direct_features: list[str],
    derived_features: list[str],
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {feature: [] for feature in direct_features + derived_features}
    for row in rows:
        for feature in direct_features:
            value = row.get(feature)
            if value is not None and math.isfinite(value):
                values[feature].append(value)
        for feature in derived_features:
            value = derived_value(feature, row)
            if value is not None and math.isfinite(value):
                values[feature].append(value)
    return {feature: feature_values for feature, feature_values in values.items() if feature_values}


def stage2_metric_id(feature: str, statistic: str) -> str:
    return f"{slugify(feature)}__{STAT_ID_MAP.get(statistic, slugify(statistic))}"


def metric_label(feature: str, statistic: str) -> str:
    return f"{STAT_LABEL_MAP.get(statistic, statistic)} {feature.replace('_', ' ')}"


def stage2_metric_facts(
    image_name: str,
    raw_path: Path,
    direct_features: list[str],
    derived_features: list[str],
    statistics: list[str],
) -> dict[str, dict[str, Any]]:
    rows = read_raw_numeric_rows(raw_path)
    by_feature = stage2_feature_values(rows, direct_features, derived_features)
    facts: dict[str, dict[str, Any]] = {}
    for feature, values in by_feature.items():
        for statistic in statistics:
            try:
                value = numeric_stat(values, statistic)
            except ValueError:
                continue
            metric_id = stage2_metric_id(feature, statistic)
            facts[metric_id] = {
                "metric": metric_id,
                "label": metric_label(feature, statistic),
                "feature": feature,
                "statistic": STAT_ID_MAP.get(statistic, statistic),
                "value": value,
                "units": "unitless" if feature in STAGE2_DERIVED_FEATURES else infer_units(feature),
                "image": image_name,
            }
    return facts


def robust_scale(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 1.0
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    if iqr > 0:
        return iqr
    std = float(np.std(arr))
    if std > 0:
        return std
    median_abs = float(abs(np.median(arr)))
    if median_abs > 0:
        return median_abs
    max_abs = float(np.max(np.abs(arr)))
    return max(max_abs, 1.0)


def metric_scales(image_facts: dict[int, dict[str, dict[str, Any]]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for facts in image_facts.values():
        for metric_id, fact in facts.items():
            grouped.setdefault(metric_id, []).append(float(fact["value"]))
    return {metric_id: robust_scale(values) for metric_id, values in grouped.items()}


def pair_metric_diffs(
    facts_a: dict[str, dict[str, Any]],
    facts_b: dict[str, dict[str, Any]],
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for metric_id in sorted(set(facts_a) & set(facts_b)):
        value_a = float(facts_a[metric_id]["value"])
        value_b = float(facts_b[metric_id]["value"])
        if math.isclose(value_a, value_b, rel_tol=1e-12, abs_tol=1e-12):
            larger = "tie"
        else:
            larger = "A" if value_a > value_b else "B"
        scale = scales.get(metric_id, 1.0)
        normalized = abs(value_a - value_b) / scale if scale else 0.0
        diffs.append(
            {
                "metric": metric_id,
                "larger": larger,
                "normalized_difference": normalized,
                "value_a": value_a,
                "value_b": value_b,
            }
        )
    diffs.sort(key=lambda item: (-float(item["normalized_difference"]), item["metric"]))
    return diffs


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


def diverse_top_diffs(diffs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for diff in diffs:
        family = metric_family(str(diff["metric"]))
        if family in used_families:
            continue
        selected.append(diff)
        used_families.add(family)
        if len(selected) >= top_k:
            return selected
    for diff in diffs:
        if diff not in selected:
            selected.append(diff)
        if len(selected) >= top_k:
            break
    return selected


def select_stage2_pairs(
    image_ids: list[int],
    image_facts: dict[int, dict[str, dict[str, Any]]],
    scales: dict[str, float],
    *,
    pair_count: int,
    pairs_per_stratum: int,
) -> list[dict[str, Any]]:
    distances: list[dict[str, Any]] = []
    for image_a, image_b in itertools.combinations(image_ids, 2):
        diffs = pair_metric_diffs(image_facts[image_a], image_facts[image_b], scales)
        if not diffs:
            continue
        distance = float(np.mean([float(item["normalized_difference"]) for item in diffs]))
        distances.append({"image_a": image_a, "image_b": image_b, "distance": distance})

    distances.sort(key=lambda item: (float(item["distance"]), item["image_a"], item["image_b"]))
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()

    def add_from(candidates: list[dict[str, Any]], stratum: str, limit: int) -> None:
        for candidate in candidates:
            key = (int(candidate["image_a"]), int(candidate["image_b"]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append({**candidate, "distance_stratum": stratum})
            if sum(item["distance_stratum"] == stratum for item in selected) >= limit:
                break

    add_from(distances, "close", pairs_per_stratum)
    midpoint = (len(distances) - 1) / 2.0
    medium_candidates = sorted(
        enumerate(distances),
        key=lambda item: (abs(item[0] - midpoint), item[1]["image_a"], item[1]["image_b"]),
    )
    add_from([item for _, item in medium_candidates], "medium", pairs_per_stratum)
    add_from(list(reversed(distances)), "large", pairs_per_stratum)

    if len(selected) < pair_count:
        add_from(list(reversed(distances)), "fill", pair_count - len(selected))

    selected.sort(
        key=lambda item: (
            {"large": 0, "medium": 1, "close": 2, "fill": 3}.get(str(item["distance_stratum"]), 9),
            -float(item["distance"]),
            item["image_a"],
            item["image_b"],
        )
    )
    return selected[:pair_count]


def topk_prompt(
    *,
    name_a: str,
    name_b: str,
    facts_a: dict[str, dict[str, Any]],
    facts_b: dict[str, dict[str, Any]],
    valid_metric_ids: list[str],
    top_k: int,
    seed: int,
) -> str:
    fact_list_a = list(facts_a.values())
    fact_list_b = list(facts_b.values())
    rng = random.Random(seed)
    rng.shuffle(fact_list_a)
    rng.shuffle(fact_list_b)
    payload = {
        "A": {"image": name_a, "metrics": fact_list_a},
        "B": {"image": name_b, "metrics": fact_list_b},
    }
    return (
        "Here are deterministic SEM powder morphology summary facts for two images as JSON. "
        "The metrics are intentionally unordered.\n"
        f"{json.dumps(payload, separators=(',', ':'))}\n"
        f"Valid metric IDs are: {json.dumps(valid_metric_ids, separators=(',', ':'))}\n"
        f"Compare image A and image B. Return JSON only: exactly {top_k} objects in an array. "
        "Each object must have keys metric, larger, and interpretation. "
        "The metric must be one of the valid metric IDs. larger must be A, B, or tie. "
        "Choose distinct morphology categories with the largest differences; do not invent metrics."
    )


def case_study_topk_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    paths = config["paths"]
    image_map = {int(k): v for k, v in read_json(paths["images_map"]).items()}
    morphology_root = REPO_ROOT / paths["morphology_root"]
    stage2 = stage2_config(config)

    image_facts: dict[int, dict[str, dict[str, Any]]] = {}
    for image_id in image_ids:
        image_name = image_map[image_id]
        raw_path = source_path(morphology_root, image_name, "_raw.csv")
        image_facts[image_id] = stage2_metric_facts(
            image_name,
            raw_path,
            list(stage2["direct_features"]),
            list(stage2["derived_features"]),
            list(stage2["statistics"]),
        )

    scales = metric_scales(image_facts)
    selected_pairs = select_stage2_pairs(
        image_ids,
        image_facts,
        scales,
        pair_count=int(stage2["pair_count"]),
        pairs_per_stratum=int(stage2["pairs_per_stratum"]),
    )

    tasks: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(selected_pairs, start=1):
        image_a = int(pair["image_a"])
        image_b = int(pair["image_b"])
        name_a = image_map[image_a]
        name_b = image_map[image_b]
        facts_a = image_facts[image_a]
        facts_b = image_facts[image_b]
        valid_metric_ids = sorted(set(facts_a) & set(facts_b))
        diffs = pair_metric_diffs(facts_a, facts_b, scales)
        diverse_truth = diverse_top_diffs(diffs, int(stage2["top_k"]))
        top_truth = [
            {
                "metric": item["metric"],
                "larger": item["larger"],
                "family": metric_family(str(item["metric"])),
                "normalized_difference": round(float(item["normalized_difference"]), 6),
                "value_a": item["value_a"],
                "value_b": item["value_b"],
            }
            for item in diverse_truth
        ]
        task_id = "__".join(["case_study_topk", f"p{pair_index:03d}"])
        tasks.append(
            {
                "task_id": task_id,
                "suite": "case_study_stage2_topk",
                "answer_type": "topk_metrics",
                "case_study_stage": 2,
                "image_a_id": image_a,
                "image_b_id": image_b,
                "image_a_name": name_a,
                "image_b_name": name_b,
                "encoding": "routed_json_topk",
                "top_k": int(stage2["top_k"]),
                "metric_vocabulary_size": len(valid_metric_ids),
                "valid_metric_ids": valid_metric_ids,
                "ground_truth": top_truth,
                "ground_truth_metrics": [item["metric"] for item in top_truth],
                "ground_truth_metric_families": [item["family"] for item in top_truth],
                "ground_truth_directions": {
                    item["metric"]: item["larger"] for item in top_truth
                },
                "pair_distance": pair["distance"],
                "distance_stratum": pair["distance_stratum"],
                "prompt": topk_prompt(
                    name_a=name_a,
                    name_b=name_b,
                    facts_a={metric_id: facts_a[metric_id] for metric_id in valid_metric_ids},
                    facts_b={metric_id: facts_b[metric_id] for metric_id in valid_metric_ids},
                    valid_metric_ids=valid_metric_ids,
                    top_k=int(stage2["top_k"]),
                    seed=int(stage2["random_seed"]) + pair_index,
                ),
                "source_paths": [
                    rel(source_path(morphology_root, name_a, "_raw.csv")),
                    rel(source_path(morphology_root, name_b, "_raw.csv")),
                ],
            }
        )
    return tasks


def case_study_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    tasks.extend(
        interface_numeric_tasks(
            config,
            image_ids,
            suite_name="case_study_stage1_numeric",
            task_prefix="case_study_numeric",
            case_study_stage=1,
        )
    )
    tasks.extend(
        interface_pairwise_tasks(
            config,
            image_ids,
            suite_name="case_study_stage1_pairwise",
            task_prefix="case_study_pairwise",
            case_study_stage=1,
        )
    )
    if config.get("case_study", {}).get("stage2_topk", {}).get("enabled", True):
        tasks.extend(case_study_topk_tasks(config, image_ids))
    return tasks


def tool_routing_config(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("tool_routing", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "pair_count": int(configured.get("pair_count", 30)),
        "top_k": int(configured.get("top_k", 3)),
    }


def tool_source_paths(
    config: dict[str, Any],
    image_ids: list[int],
) -> list[str]:
    names = tr.image_map(config)
    root = tr.morphology_root(config)
    return [rel(tr.source_path(root, names[int(image_id)], "_raw.csv")) for image_id in image_ids]


def tool_task_record(
    *,
    task_id: str,
    task_type: str,
    paraphrase_id: str,
    paraphrase_family: str,
    user_request: str,
    prompt: str,
    expected_tool: str,
    expected_arguments: dict[str, Any],
    ground_truth: Any,
    source_paths: list[str],
    accepted_arguments: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": task_id,
        "suite": "tool_routing",
        "answer_type": "tool_call",
        "encoding": "tool_schema_json",
        "tool_scoring_policy": (
            "strict expected_arguments plus declared accepted_arguments; accepted routes are "
            "included only when deterministic execution preserves the primary answer"
        ),
        "tool_task_type": task_type,
        "paraphrase_id": paraphrase_id,
        "paraphrase_family": paraphrase_family,
        "user_request": user_request,
        "expected_tool": expected_tool,
        "expected_arguments": expected_arguments,
        "accepted_arguments": accepted_arguments or [],
        "ground_truth": ground_truth,
        "prompt": prompt,
        "source_paths": source_paths,
    }
    if extra:
        record.update(extra)
    return record


def current_pair_context(
    image_a: int,
    image_b: int,
    name_a: str,
    name_b: str,
) -> dict[str, dict[str, Any]]:
    return {
        "A": {"image_id": int(image_a), "image_name": name_a},
        "B": {"image_id": int(image_b), "image_name": name_b},
    }


def argument_key(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def add_median_percentile50_variant(
    arguments: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    statistic = arguments.get("statistic")
    if statistic not in {"median", "percentile_50"}:
        return
    swapped = dict(arguments)
    swapped["statistic"] = "percentile_50" if statistic == "median" else "median"
    candidates.append(swapped)


def accepted_equivalent_arguments(
    *,
    config: dict[str, Any],
    tool: str,
    expected_arguments: dict[str, Any],
    raw_equivalents: list[dict[str, Any]] | None,
    truth: Any,
    task_type: str,
    scale_image_ids: list[int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(arguments: dict[str, Any]) -> None:
        full = {**expected_arguments, **arguments}
        candidates.append(full)
        add_median_percentile50_variant(full, candidates)

    add(expected_arguments)
    for equivalent in raw_equivalents or []:
        add(dict(equivalent))

    expected_key = argument_key(expected_arguments)
    accepted: list[dict[str, Any]] = []
    seen = {expected_key}
    for candidate in candidates:
        key = argument_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = tr.execute_normalized_tool_call(
                config,
                tool,
                candidate,
                scale_image_ids=[int(item) for item in scale_image_ids],
            )
        except Exception:
            continue
        if tr.result_matches(result, truth, task_type):
            accepted.append(candidate)
    return accepted


def tool_routing_single_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    names = tr.image_map(config)
    registry = tr.image_registry(config, image_ids)
    tasks: list[dict[str, Any]] = []
    for image_id, intent in itertools.product(image_ids, tr.SINGLE_INTENTS):
        image_name = names[int(image_id)]
        expected_arguments = {
            "image_id": int(image_id),
            "feature": intent["feature"],
            "statistic": intent["statistic"],
        }
        truth = tr.execute_normalized_tool_call(
            config,
            tr.GET_STAT_TOOL,
            expected_arguments,
            scale_image_ids=image_ids,
        )
        accepted_arguments = accepted_equivalent_arguments(
            config=config,
            tool=tr.GET_STAT_TOOL,
            expected_arguments=expected_arguments,
            raw_equivalents=intent.get("accepted_equivalents", []),
            truth=truth,
            task_type="single_stat",
            scale_image_ids=image_ids,
        )
        for paraphrase_index, template in enumerate(intent["paraphrases"], start=1):
            paraphrase_id = f"p{paraphrase_index:02d}"
            user_request = template.format(image_id=int(image_id), image_name=image_name)
            task_id = "__".join(
                [
                    "tool_routing_single",
                    f"img{int(image_id):02d}",
                    slugify(str(intent["key"])),
                    paraphrase_id,
                ]
            )
            tasks.append(
                tool_task_record(
                    task_id=task_id,
                    task_type="single_stat",
                    paraphrase_id=paraphrase_id,
                    paraphrase_family=str(intent["paraphrase_family"]),
                    user_request=user_request,
                    prompt=tr.tool_routing_prompt(
                        user_request=user_request,
                        image_registry=registry,
                        task_type="single_stat",
                        current_image={"image_id": int(image_id), "image_name": image_name},
                    ),
                    expected_tool=tr.GET_STAT_TOOL,
                    expected_arguments=expected_arguments,
                    accepted_arguments=accepted_arguments,
                    ground_truth=truth,
                    source_paths=tool_source_paths(config, [int(image_id)]),
                    extra={
                        "image_id": int(image_id),
                        "image_name": image_name,
                        "feature": intent["feature"],
                        "statistic": intent["statistic"],
                        "tool_intent_key": intent["key"],
                    },
                )
            )
    return tasks


def tool_routing_pairwise_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    names = tr.image_map(config)
    registry = tr.image_registry(config, image_ids)
    pair_count = int(tool_routing_config(config)["pair_count"])
    image_pairs = list(itertools.combinations(image_ids, 2))[:pair_count]
    tasks: list[dict[str, Any]] = []
    for pair_index, (image_a, image_b) in enumerate(image_pairs, start=1):
        name_a = names[int(image_a)]
        name_b = names[int(image_b)]
        for intent in tr.PAIRWISE_INTENTS:
            expected_arguments = {
                "image_a_id": int(image_a),
                "image_b_id": int(image_b),
                "feature": intent["feature"],
                "statistic": intent["statistic"],
                "relation": intent["relation"],
            }
            truth = tr.execute_normalized_tool_call(
                config,
                tr.COMPARE_STAT_TOOL,
                expected_arguments,
                scale_image_ids=image_ids,
            )
            accepted_arguments = accepted_equivalent_arguments(
                config=config,
                tool=tr.COMPARE_STAT_TOOL,
                expected_arguments=expected_arguments,
                raw_equivalents=intent.get("accepted_equivalents", []),
                truth=truth,
                task_type="pairwise_compare",
                scale_image_ids=image_ids,
            )
            for paraphrase_index, template in enumerate(intent["paraphrases"], start=1):
                paraphrase_id = f"p{paraphrase_index:02d}"
                user_request = template.format(
                    image_a_id=int(image_a),
                    image_b_id=int(image_b),
                    name_a=name_a,
                    name_b=name_b,
                )
                task_id = "__".join(
                    [
                        "tool_routing_pairwise",
                        f"p{pair_index:03d}",
                        slugify(str(intent["key"])),
                        paraphrase_id,
                    ]
                )
                tasks.append(
                    tool_task_record(
                        task_id=task_id,
                        task_type="pairwise_compare",
                        paraphrase_id=paraphrase_id,
                        paraphrase_family=str(intent["paraphrase_family"]),
                        user_request=user_request,
                    prompt=tr.tool_routing_prompt(
                        user_request=user_request,
                        image_registry=registry,
                        task_type="pairwise_compare",
                        current_pair=current_pair_context(
                            int(image_a),
                            int(image_b),
                            name_a,
                            name_b,
                        ),
                    ),
                    expected_tool=tr.COMPARE_STAT_TOOL,
                    expected_arguments=expected_arguments,
                        accepted_arguments=accepted_arguments,
                        ground_truth=truth,
                        source_paths=tool_source_paths(config, [int(image_a), int(image_b)]),
                        extra={
                            "image_a_id": int(image_a),
                            "image_b_id": int(image_b),
                            "image_a_name": name_a,
                            "image_b_name": name_b,
                            "feature": intent["feature"],
                            "statistic": intent["statistic"],
                            "tool_relation": intent["relation"],
                            "tool_intent_key": intent["key"],
                        },
                    )
                )
    return tasks


def tool_routing_topk_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    names = tr.image_map(config)
    registry = tr.image_registry(config, image_ids)
    routing = tool_routing_config(config)
    pair_count = int(routing["pair_count"])
    top_k = int(routing["top_k"])
    image_pairs = list(itertools.combinations(image_ids, 2))[:pair_count]
    tasks: list[dict[str, Any]] = []
    for pair_index, (image_a, image_b) in enumerate(image_pairs, start=1):
        name_a = names[int(image_a)]
        name_b = names[int(image_b)]
        expected_arguments = {
            "image_a_id": int(image_a),
            "image_b_id": int(image_b),
            "top_k": top_k,
            "metric_scope": "expanded_morphology",
        }
        truth = tr.execute_normalized_tool_call(
            config,
            tr.TOP_DIFFS_TOOL,
            expected_arguments,
            scale_image_ids=image_ids,
        )
        for paraphrase_index, paraphrase in enumerate(tr.TOPK_PARAPHRASES, start=1):
            paraphrase_id = f"p{paraphrase_index:02d}"
            user_request = paraphrase["text"].format(
                image_a_id=int(image_a),
                image_b_id=int(image_b),
                name_a=name_a,
                name_b=name_b,
            )
            task_id = "__".join(
                [
                    "tool_routing_topk",
                    f"p{pair_index:03d}",
                    slugify(str(paraphrase["key"])),
                    paraphrase_id,
                ]
            )
            tasks.append(
                tool_task_record(
                    task_id=task_id,
                    task_type="topk_differences",
                    paraphrase_id=paraphrase_id,
                    paraphrase_family=str(paraphrase["paraphrase_family"]),
                    user_request=user_request,
                    prompt=tr.tool_routing_prompt(
                        user_request=user_request,
                        image_registry=registry,
                        task_type="topk_differences",
                        current_pair=current_pair_context(
                            int(image_a),
                            int(image_b),
                            name_a,
                            name_b,
                        ),
                    ),
                    expected_tool=tr.TOP_DIFFS_TOOL,
                    expected_arguments=expected_arguments,
                    ground_truth=truth,
                    source_paths=tool_source_paths(config, [int(image_a), int(image_b)]),
                    extra={
                        "image_a_id": int(image_a),
                        "image_b_id": int(image_b),
                        "image_a_name": name_a,
                        "image_b_name": name_b,
                        "top_k": top_k,
                        "scale_image_ids": [int(item) for item in image_ids],
                        "tool_intent_key": paraphrase["key"],
                    },
                )
            )
    return tasks


def tool_routing_tasks(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    if not tool_routing_config(config)["enabled"]:
        return []
    tasks: list[dict[str, Any]] = []
    tasks.extend(tool_routing_single_tasks(config, image_ids))
    tasks.extend(tool_routing_pairwise_tasks(config, image_ids))
    tasks.extend(tool_routing_topk_tasks(config, image_ids))
    return tasks


def build_tasks(config: dict[str, Any], suite: str, image_limit: int | None) -> list[dict[str, Any]]:
    image_ids = selected_image_ids(config, image_limit)
    tasks: list[dict[str, Any]] = []
    if suite in {"core", "all"} and config.get("core_benchmark", {}).get("enabled", True):
        tasks.extend(core_tasks(config, image_ids))
    if suite in {"interface", "all"} and config.get("interface_suite", {}).get("enabled", True):
        tasks.extend(interface_numeric_tasks(config, image_ids))
        tasks.extend(interface_pairwise_tasks(config, image_ids))
    if suite == "case_study" and config.get("case_study", {}).get("enabled", True):
        tasks.extend(case_study_tasks(config, image_ids))
    if suite == "tool_routing" and config.get("tool_routing", {}).get("enabled", True):
        tasks.extend(tool_routing_tasks(config, image_ids))
    return tasks


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    image_limit = args.image_limit or int(config["image_policy"]["default_limit"])
    tasks = build_tasks(config, args.suite, image_limit)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output = args.output
    if output is None:
        suffix = args.suite if args.suite != "all" else "tasks"
        output = f"{config['paths']['manifests_dir']}/{suffix}_image{image_limit}.jsonl"

    count = write_jsonl(output, tasks)
    print(f"Wrote {count:,} tasks to {output}")


if __name__ == "__main__":
    main()
