#!/usr/bin/env python3
"""Shared tool-routing definitions for AI4Mat characterization experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_utils import REPO_ROOT, read_json, selected_image_ids, slugify


GET_STAT_TOOL = "get_morphology_statistic"
COMPARE_STAT_TOOL = "compare_morphology_statistic"
TOP_DIFFS_TOOL = "top_morphology_differences"

PRIMARY_FEATURES = [
    "equivalent_diameter",
    "circularity",
    "aspect_ratio",
    "elongation",
]

PRIMARY_STATISTICS = [
    "percentile_10",
    "percentile_50",
    "percentile_90",
    "median",
    "d90_minus_d10_over_d50",
]

TOP_DIRECT_FEATURES = [
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

TOP_DERIVED_FEATURES = [
    "aspect_ratio",
    "elongation",
    "convexity_deficit",
    "bounding_box_fill",
    "perimeter_to_diameter_ratio",
]

TOP_STATISTICS = [
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

FEATURE_ALIASES = {
    "diameter": "equivalent_diameter",
    "equiv_diameter": "equivalent_diameter",
    "equivalent diameter": "equivalent_diameter",
    "particle_size": "equivalent_diameter",
    "particle size": "equivalent_diameter",
    "size": "equivalent_diameter",
    "roundness": "circularity",
    "sphericity": "circularity",
    "spherical": "circularity",
    "aspect ratio": "aspect_ratio",
    "elongated": "aspect_ratio",
    "elongation": "elongation",
}

STATISTIC_ALIASES = {
    "d10": "percentile_10",
    "p10": "percentile_10",
    "percentile10": "percentile_10",
    "10th_percentile": "percentile_10",
    "d50": "percentile_50",
    "p50": "percentile_50",
    "percentile50": "percentile_50",
    "median": "median",
    "typical": "percentile_50",
    "d90": "percentile_90",
    "p90": "percentile_90",
    "percentile90": "percentile_90",
    "90th_percentile": "percentile_90",
    "span": "d90_minus_d10_over_d50",
    "distribution_span": "d90_minus_d10_over_d50",
    "d90_d10_d50": "d90_minus_d10_over_d50",
    "d90_minus_d10_over_d50": "d90_minus_d10_over_d50",
}

RELATION_ALIASES = {
    "greater": "greater",
    "larger": "greater",
    "higher": "greater",
    "more": "greater",
    "bigger": "greater",
    "maximum": "greater",
    "max": "greater",
    "lower": "lower",
    "less": "lower",
    "fewer": "lower",
    "smaller": "lower",
    "minimum": "lower",
    "min": "lower",
}

TOOL_ALIASES = {
    GET_STAT_TOOL: GET_STAT_TOOL,
    "get_statistic": GET_STAT_TOOL,
    "morphology_statistic": GET_STAT_TOOL,
    COMPARE_STAT_TOOL: COMPARE_STAT_TOOL,
    "compare_statistic": COMPARE_STAT_TOOL,
    "compare_morphology": COMPARE_STAT_TOOL,
    TOP_DIFFS_TOOL: TOP_DIFFS_TOOL,
    "top_differences": TOP_DIFFS_TOOL,
    "top_morphology_metrics": TOP_DIFFS_TOOL,
}

JARGON_MAPPINGS = [
    "rounder / more spherical / more circular -> feature circularity; larger means more round",
    "elongated / less spherical / needle-like -> feature aspect_ratio or elongation; larger means more elongated",
    "fewer elongated particles -> feature aspect_ratio or elongation with relation lower",
    "bigger chunks / coarser particles / top tail / largest particles -> feature equivalent_diameter, statistic percentile_90",
    "typical size / middle particle size / D50 -> feature equivalent_diameter, statistic percentile_50",
    "fine tail / small particles / D10 -> feature equivalent_diameter, statistic percentile_10",
    "spread / width / broad distribution / polydisperse -> statistic d90_minus_d10_over_d50",
]

ROUTING_POLICIES = [
    "All benchmark requests are answerable from the provided schema; do not return clarify, none, or a prose refusal.",
    "For pairwise and top-k requests, A and B are defined by current_pair; use those exact image IDs.",
    "median and percentile_50 are the same deterministic statistic; prefer the statistic named by the request.",
    "For elongated-particle language, aspect_ratio and elongation are accepted shape proxies when they preserve the deterministic comparison.",
    "The LLM selects the tool and arguments only; deterministic code computes all morphology values.",
]

SINGLE_INTENTS = [
    {
        "key": "fine_tail_d10",
        "feature": "equivalent_diameter",
        "statistic": "percentile_10",
        "paraphrase_family": "fine_tail",
        "paraphrases": [
            "For {image_name}, route a tool call to report the D10 equivalent diameter.",
            "For image {image_id}, which tool call gets the fine-tail particle size?",
            "I need the small-particle tail for {image_name}; use the schema to request it.",
        ],
    },
    {
        "key": "typical_size_d50",
        "feature": "equivalent_diameter",
        "statistic": "percentile_50",
        "paraphrase_family": "typical_size",
        "paraphrases": [
            "For {image_name}, route a tool call for the typical particle size.",
            "For image {image_id}, request the D50 equivalent diameter.",
            "I need the middle particle size for {image_name}; emit the correct tool call.",
        ],
    },
    {
        "key": "coarse_tail_d90",
        "feature": "equivalent_diameter",
        "statistic": "percentile_90",
        "paraphrase_family": "coarse_tail",
        "paraphrases": [
            "For {image_name}, route a tool call for the bigger chunks in the top tail.",
            "For image {image_id}, request the D90 equivalent diameter.",
            "I need the coarse-particle tail for {image_name}; emit the correct tool call.",
        ],
    },
    {
        "key": "roundness_median",
        "feature": "circularity",
        "statistic": "median",
        "paraphrase_family": "roundness",
        "paraphrases": [
            "For {image_name}, route a tool call for median particle roundness.",
            "For image {image_id}, request the median circularity.",
            "How spherical are particles in {image_name}? Emit the matching tool call.",
        ],
    },
    {
        "key": "distribution_width_span",
        "feature": "equivalent_diameter",
        "statistic": "d90_minus_d10_over_d50",
        "paraphrase_family": "distribution_width",
        "paraphrases": [
            "For {image_name}, route a tool call for particle-size distribution width.",
            "For image {image_id}, request the normalized D90-D10 over D50 size span.",
            "I need how polydisperse {image_name} is; emit the matching tool call.",
        ],
    },
]

PAIRWISE_INTENTS = [
    {
        "key": "bigger_chunks",
        "feature": "equivalent_diameter",
        "statistic": "percentile_90",
        "relation": "greater",
        "paraphrase_family": "coarse_tail",
        "paraphrases": [
            "Between {name_a} and {name_b}, which powder has bigger chunks?",
            "Compare images {image_a_id} and {image_b_id}: which has the larger top-tail particle size?",
            (
                "For A={name_a} (image {image_a_id}) and B={name_b} "
                "(image {image_b_id}), route a tool call to decide which has coarser D90 particles."
            ),
        ],
    },
    {
        "key": "fewer_elongated",
        "feature": "aspect_ratio",
        "statistic": "median",
        "relation": "lower",
        "accepted_equivalents": [
            {"feature": "elongation", "statistic": "median", "relation": "lower"}
        ],
        "paraphrase_family": "elongation",
        "paraphrases": [
            "Between {name_a} and {name_b}, which powder has fewer elongated particles?",
            "Compare images {image_a_id} and {image_b_id}: which batch is less needle-like?",
            (
                "For A={name_a} (image {image_a_id}) and B={name_b} "
                "(image {image_b_id}), route a tool call to identify which has lower median elongation."
            ),
        ],
    },
    {
        "key": "broader_distribution",
        "feature": "equivalent_diameter",
        "statistic": "d90_minus_d10_over_d50",
        "relation": "greater",
        "paraphrase_family": "distribution_width",
        "paraphrases": [
            "Between {name_a} and {name_b}, which powder has the broader particle-size spread?",
            "Compare images {image_a_id} and {image_b_id}: which has the wider equivalent-diameter distribution?",
            (
                "For A={name_a} (image {image_a_id}) and B={name_b} "
                "(image {image_b_id}), route a tool call to decide which is more polydisperse."
            ),
        ],
    },
]

TOPK_PARAPHRASES = [
    {
        "key": "top_three_differences",
        "paraphrase_family": "open_comparison",
        "text": (
            "For A={name_a} (image {image_a_id}) versus B={name_b} (image {image_b_id}), "
            "route a tool call to find the top 3 morphology differences across the expanded "
            "characterization vocabulary."
        ),
    },
    {
        "key": "most_separating_metrics",
        "paraphrase_family": "open_comparison",
        "text": (
            "Compare A=image {image_a_id} ({name_a}) and B=image {image_b_id} ({name_b}); "
            "which three characterization metrics most separate these powders? Return the tool call only."
        ),
    },
]


def image_map(config: dict[str, Any]) -> dict[int, str]:
    return {int(key): value for key, value in read_json(config["paths"]["images_map"]).items()}


def morphology_root(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config["paths"]["morphology_root"]


def source_path(root: Path, image_name: str, suffix: str) -> Path:
    return root / image_name / f"{image_name}{suffix}"


def infer_units(feature: str) -> str:
    if feature in {
        "eccentricity",
        "solidity",
        "extent",
        "circularity",
        "aspect_ratio",
        "elongation",
        "convexity_deficit",
        "bounding_box_fill",
        "perimeter_to_diameter_ratio",
    }:
        return "unitless"
    if "area" in feature:
        return "pixels_squared"
    return "pixels"


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
    if statistic in {"median", "percentile_50"}:
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
    if statistic == "coefficient_of_variation":
        mean = float(np.mean(arr))
        if mean == 0:
            raise ValueError("Cannot calculate coefficient of variation with mean=0")
        return float(np.std(arr) / abs(mean))
    raise KeyError(f"Unsupported statistic: {statistic}")


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


def feature_values_for_image(config: dict[str, Any], image_id: int, feature: str) -> list[float]:
    names = image_map(config)
    root = morphology_root(config)
    raw_path = source_path(root, names[int(image_id)], "_raw.csv")
    rows = read_raw_numeric_rows(raw_path)
    values: list[float] = []
    for row in rows:
        if feature in row:
            value = row[feature]
        else:
            value = derived_value(feature, row)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError(f"No values for feature {feature!r} in image {image_id}")
    return values


def morphology_statistic(
    config: dict[str, Any],
    image_id: int,
    feature: str,
    statistic: str,
) -> dict[str, Any]:
    value = numeric_stat(feature_values_for_image(config, int(image_id), feature), statistic)
    return {
        "value": value,
        "units": infer_units(feature),
        "image_id": int(image_id),
        "feature": feature,
        "statistic": statistic,
    }


def compare_morphology_statistic(
    config: dict[str, Any],
    image_a_id: int,
    image_b_id: int,
    feature: str,
    statistic: str,
    relation: str,
) -> dict[str, Any]:
    relation = normalize_relation(relation) or "greater"
    value_a = morphology_statistic(config, int(image_a_id), feature, statistic)["value"]
    value_b = morphology_statistic(config, int(image_b_id), feature, statistic)["value"]
    if math.isclose(float(value_a), float(value_b), rel_tol=1e-12, abs_tol=1e-12):
        label = "tie"
    elif relation == "lower":
        label = "A" if float(value_a) < float(value_b) else "B"
    else:
        label = "A" if float(value_a) > float(value_b) else "B"
    return {
        "label": label,
        "value_a": value_a,
        "value_b": value_b,
        "image_a_id": int(image_a_id),
        "image_b_id": int(image_b_id),
        "feature": feature,
        "statistic": statistic,
        "relation": relation,
    }


def top_metric_id(feature: str, statistic: str) -> str:
    return f"{slugify(feature)}__{STAT_ID_MAP.get(statistic, slugify(statistic))}"


def stage2_feature_values(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {feature: [] for feature in TOP_DIRECT_FEATURES + TOP_DERIVED_FEATURES}
    for row in rows:
        for feature in TOP_DIRECT_FEATURES:
            value = row.get(feature)
            if value is not None and math.isfinite(value):
                values[feature].append(value)
        for feature in TOP_DERIVED_FEATURES:
            value = derived_value(feature, row)
            if value is not None and math.isfinite(value):
                values[feature].append(value)
    return {feature: feature_values for feature, feature_values in values.items() if feature_values}


def stage2_metric_facts(
    config: dict[str, Any],
    image_id: int,
) -> dict[str, dict[str, Any]]:
    names = image_map(config)
    root = morphology_root(config)
    raw_path = source_path(root, names[int(image_id)], "_raw.csv")
    rows = read_raw_numeric_rows(raw_path)
    by_feature = stage2_feature_values(rows)
    facts: dict[str, dict[str, Any]] = {}
    for feature, values in by_feature.items():
        for statistic in TOP_STATISTICS:
            try:
                value = numeric_stat(values, statistic)
            except ValueError:
                continue
            metric_id = top_metric_id(feature, statistic)
            facts[metric_id] = {
                "metric": metric_id,
                "feature": feature,
                "statistic": STAT_ID_MAP.get(statistic, statistic),
                "value": value,
                "units": infer_units(feature),
                "image_id": int(image_id),
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
                "family": metric_family(metric_id),
                "normalized_difference": normalized,
                "value_a": value_a,
                "value_b": value_b,
            }
        )
    diffs.sort(key=lambda item: (-float(item["normalized_difference"]), item["metric"]))
    return diffs


def diverse_top_diffs(diffs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for diff in diffs:
        if diff["family"] in used_families:
            continue
        selected.append(diff)
        used_families.add(str(diff["family"]))
        if len(selected) >= top_k:
            return selected
    for diff in diffs:
        if diff not in selected:
            selected.append(diff)
        if len(selected) >= top_k:
            break
    return selected


def top_morphology_differences(
    config: dict[str, Any],
    image_a_id: int,
    image_b_id: int,
    top_k: int,
    scale_image_ids: list[int] | None = None,
) -> dict[str, Any]:
    if scale_image_ids is None:
        scale_image_ids = selected_image_ids(config)
    relevant_image_ids = sorted({int(image_a_id), int(image_b_id), *[int(item) for item in scale_image_ids]})
    image_facts = {image_id: stage2_metric_facts(config, image_id) for image_id in relevant_image_ids}
    scales = metric_scales({image_id: image_facts[image_id] for image_id in scale_image_ids})
    diffs = pair_metric_diffs(image_facts[int(image_a_id)], image_facts[int(image_b_id)], scales)
    top = diverse_top_diffs(diffs, int(top_k))
    return {
        "top_k": int(top_k),
        "image_a_id": int(image_a_id),
        "image_b_id": int(image_b_id),
        "differences": [
            {
                "metric": item["metric"],
                "larger": item["larger"],
                "family": item["family"],
                "normalized_difference": round(float(item["normalized_difference"]), 6),
            }
            for item in top
        ],
    }


def normalize_identifier(value: Any, valid_ids: list[str], aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    aliases = aliases or {}
    valid_lookup = {item.lower(): item for item in valid_ids}
    cleaned = str(value).strip().lower()
    if cleaned in valid_lookup:
        return valid_lookup[cleaned]
    if cleaned in aliases:
        return aliases[cleaned]
    slug = slugify(cleaned)
    if slug in valid_lookup:
        return valid_lookup[slug]
    return aliases.get(slug)


def normalize_relation(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in RELATION_ALIASES:
        return RELATION_ALIASES[cleaned]
    return RELATION_ALIASES.get(slugify(cleaned))


def normalize_tool_name(value: Any) -> str | None:
    return normalize_identifier(value, [GET_STAT_TOOL, COMPARE_STAT_TOOL, TOP_DIFFS_TOOL], TOOL_ALIASES)


def normalize_image_reference(value: Any, names: dict[int, str]) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in names else None
    if isinstance(value, float) and value.is_integer():
        number = int(value)
        return number if number in names else None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        number = int(cleaned)
        return number if number in names else None
    match = re.fullmatch(r"img(?:age)?[_\s-]*(\d+)", cleaned.lower())
    if match:
        number = int(match.group(1))
        return number if number in names else None
    lookup = {name.lower(): image_id for image_id, name in names.items()}
    lookup.update({slugify(name): image_id for image_id, name in names.items()})
    return lookup.get(cleaned.lower(), lookup.get(slugify(cleaned)))


def normalize_tool_call_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    for key in ["tool_call", "function_call", "call", "answer"]:
        value = payload.get(key)
        if isinstance(value, dict):
            nested = normalize_tool_call_payload(value)
            if nested is not None:
                return nested
    tool = normalize_tool_name(payload.get("tool", payload.get("name", payload.get("function"))))
    arguments = payload.get("arguments", payload.get("args", payload.get("parameters")))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if tool is None or not isinstance(arguments, dict):
        return None
    return {"tool": tool, "arguments": arguments}


def normalize_tool_arguments(
    tool: str,
    arguments: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    names = image_map(config)
    if tool == GET_STAT_TOOL:
        image_id = normalize_image_reference(arguments.get("image_id", arguments.get("image")), names)
        feature = normalize_identifier(arguments.get("feature"), PRIMARY_FEATURES, FEATURE_ALIASES)
        statistic = normalize_identifier(arguments.get("statistic"), PRIMARY_STATISTICS, STATISTIC_ALIASES)
        normalized = {"image_id": image_id, "feature": feature, "statistic": statistic}
        missing = [key for key, value in normalized.items() if value is None]
        return normalized, f"missing_or_invalid_{','.join(missing)}" if missing else None
    if tool == COMPARE_STAT_TOOL:
        image_a_id = normalize_image_reference(
            arguments.get("image_a_id", arguments.get("image_a", arguments.get("A"))),
            names,
        )
        image_b_id = normalize_image_reference(
            arguments.get("image_b_id", arguments.get("image_b", arguments.get("B"))),
            names,
        )
        feature = normalize_identifier(arguments.get("feature"), PRIMARY_FEATURES, FEATURE_ALIASES)
        statistic = normalize_identifier(arguments.get("statistic"), PRIMARY_STATISTICS, STATISTIC_ALIASES)
        relation = normalize_relation(arguments.get("relation", "greater"))
        normalized = {
            "image_a_id": image_a_id,
            "image_b_id": image_b_id,
            "feature": feature,
            "statistic": statistic,
            "relation": relation,
        }
        missing = [key for key, value in normalized.items() if value is None]
        return normalized, f"missing_or_invalid_{','.join(missing)}" if missing else None
    if tool == TOP_DIFFS_TOOL:
        image_a_id = normalize_image_reference(
            arguments.get("image_a_id", arguments.get("image_a", arguments.get("A"))),
            names,
        )
        image_b_id = normalize_image_reference(
            arguments.get("image_b_id", arguments.get("image_b", arguments.get("B"))),
            names,
        )
        top_k_value = arguments.get("top_k", arguments.get("k", 3))
        try:
            top_k = int(top_k_value)
        except (TypeError, ValueError):
            top_k = None
        metric_scope = str(arguments.get("metric_scope", "expanded_morphology")).strip().lower()
        normalized = {
            "image_a_id": image_a_id,
            "image_b_id": image_b_id,
            "top_k": top_k,
            "metric_scope": slugify(metric_scope),
        }
        missing = [key for key, value in normalized.items() if value is None]
        return normalized, f"missing_or_invalid_{','.join(missing)}" if missing else None
    return {}, "unknown_tool"


def execute_normalized_tool_call(
    config: dict[str, Any],
    tool: str,
    arguments: dict[str, Any],
    *,
    scale_image_ids: list[int] | None = None,
) -> dict[str, Any]:
    if tool == GET_STAT_TOOL:
        return morphology_statistic(
            config,
            int(arguments["image_id"]),
            str(arguments["feature"]),
            str(arguments["statistic"]),
        )
    if tool == COMPARE_STAT_TOOL:
        return compare_morphology_statistic(
            config,
            int(arguments["image_a_id"]),
            int(arguments["image_b_id"]),
            str(arguments["feature"]),
            str(arguments["statistic"]),
            str(arguments["relation"]),
        )
    if tool == TOP_DIFFS_TOOL:
        return top_morphology_differences(
            config,
            int(arguments["image_a_id"]),
            int(arguments["image_b_id"]),
            int(arguments["top_k"]),
            scale_image_ids=scale_image_ids,
        )
    raise KeyError(f"Unsupported tool: {tool}")


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "tool": GET_STAT_TOOL,
            "arguments": {
                "image_id": "integer",
                "feature": PRIMARY_FEATURES,
                "statistic": PRIMARY_STATISTICS,
            },
        },
        {
            "tool": COMPARE_STAT_TOOL,
            "arguments": {
                "image_a_id": "integer",
                "image_b_id": "integer",
                "feature": PRIMARY_FEATURES,
                "statistic": PRIMARY_STATISTICS,
                "relation": ["greater", "lower"],
            },
        },
        {
            "tool": TOP_DIFFS_TOOL,
            "arguments": {
                "image_a_id": "integer",
                "image_b_id": "integer",
                "top_k": 3,
                "metric_scope": ["expanded_morphology"],
            },
        },
    ]


def tool_routing_prompt(
    *,
    user_request: str,
    image_registry: list[dict[str, Any]],
    task_type: str,
    current_image: dict[str, Any] | None = None,
    current_pair: dict[str, Any] | None = None,
) -> str:
    payload = {
        "task_type": task_type,
        "tool_schemas": tool_schemas(),
        "allowed_images": image_registry,
        "current_image": current_image,
        "current_pair": current_pair,
        "materials_jargon_mapping": JARGON_MAPPINGS,
        "routing_policies": ROUTING_POLICIES,
        "allowed_topk_metric_features": TOP_DIRECT_FEATURES + TOP_DERIVED_FEATURES,
        "allowed_topk_statistics": TOP_STATISTICS,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return (
        "You are routing a natural-language SEM powder characterization request to one "
        "deterministic tool. Do not calculate morphology values yourself.\n"
        f"{json.dumps(payload, separators=(',', ':'))}\n"
        "Return JSON only. Return exactly one object with keys tool and arguments. "
        "Use only allowed IDs from the schemas and current task context. "
        "Do not include prose, markdown, clarification requests, or computed values.\n"
        f"User request: {user_request}"
    )


def image_registry(config: dict[str, Any], image_ids: list[int]) -> list[dict[str, Any]]:
    names = image_map(config)
    return [{"image_id": int(image_id), "image_name": names[int(image_id)]} for image_id in image_ids]


def result_matches(predicted: dict[str, Any], truth: Any, task_type: str) -> bool:
    if task_type == "single_stat":
        try:
            prediction = float(predicted["value"])
            expected = float(truth["value"] if isinstance(truth, dict) else truth)
        except (KeyError, TypeError, ValueError):
            return False
        if expected == 0:
            return prediction == 0
        return abs((prediction - expected) / expected) * 100.0 <= 1.0
    if task_type == "pairwise_compare":
        expected_label = truth.get("label") if isinstance(truth, dict) else truth
        return predicted.get("label") == expected_label
    if task_type == "topk_differences":
        predicted_items = predicted.get("differences", [])
        truth_items = truth.get("differences", []) if isinstance(truth, dict) else truth
        if not isinstance(predicted_items, list) or not isinstance(truth_items, list):
            return False
        if len(predicted_items) != len(truth_items):
            return False
        for predicted_item, truth_item in zip(predicted_items, truth_items):
            if predicted_item.get("metric") != truth_item.get("metric"):
                return False
            if predicted_item.get("larger") != truth_item.get("larger"):
                return False
        return True
    return False
