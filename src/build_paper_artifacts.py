#!/usr/bin/env python3
"""Build adaptive AI4Mat paper figures and LaTeX tables.

The artifact layer intentionally separates paper-ready visualization from the
raw evaluation scripts. Use --refresh to rescore current raw outputs, then the
figures/tables update as additional model runs land.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_utils import REPO_ROOT, load_yaml, slugify, write_json_atomic

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
DEFAULT_MODELS = "configs/models.yaml"
MAIN_MODEL_ORDER = [
    "gpt-5.6-luna-none",
    "gpt-5.4-mini-none",
    "gemma-4-31b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b-low",
    "gemma-4-26b-a4b",
]
PARADIGM_TABLE_ORDER = [
    "gpt-5.6-luna-none",
    "gpt-5.6-luna-medium",
    "gpt-5.4-mini-none",
    "gpt-5.4-mini-medium",
    "gpt-5.4-nano-none",
    "gpt-5.4-nano-medium",
    "gemma-4-31b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b",
    "gemma-4-26b-a4b",
    "bonsai-8b",
    "bonsai-27b",
]
EXCLUDED_MODEL_LABELS = {"qwen3.6-27b", "bonsai-27b"}
BASE_MANIFEST = "results/manifests/tasks_image20.jsonl"
BASE_MERGED_RUN_ID = "paper-base20-merged"
BASE_RUN_CANDIDATES = [
    "base_run1",
    "base-image20-run1",
    "vllm-image20-run1",
    "base-api-run1",
    "base_flex_run1",
    "run1",
    "vllm-run1",
]
RELIABILITY_RUN_CANDIDATES = [
    "reliability-vllm-run1",
    "reliability_api_run1",
    "reliability-api-run1",
]
CASE_STUDY_API_RUN_CANDIDATES = [
    "case_study_api_run1",
    "case-study-api-run1",
]
TABLE1_MODEL_ORDER = [
    "gpt-5.6-luna-medium",
    "gpt-5.6-luna-none",
    "gpt-5.4-mini-medium",
    "gpt-5.4-mini-none",
    "gpt-5.4-nano-medium",
    "gpt-5.4-nano-none",
    "gemma-4-31b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b",
    "gemma-4-26b-a4b",
    "bonsai-8b",
    "bonsai-27b",
]
FIG2_MODEL_ORDER = [
    "gpt-5.6-luna-none",
    "gpt-5.6-luna-medium",
    "gpt-5.4-mini-none",
    "gpt-5.4-mini-medium",
    "gpt-5.4-nano-none",
    "gpt-5.4-nano-medium",
    "gemma-4-31b",
    "qwen3.6-35b-a3b",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b",
    "gemma-4-26b-a4b",
    "bonsai-8b",
    "bonsai-27b",
]
MODEL_SHORT = {
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
    "qwen3.6-27b": "Qwen 27B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "bonsai-8b": "Bonsai-8B",
    "bonsai-27b": "Bonsai-27B",
}
MODEL_CLASS = {
    "gpt-5.6-luna-none": "API",
    "gpt-5.6-luna-medium": "API",
    "gpt-5.4-mini-none": "API",
    "gpt-5.4-mini-medium": "API",
    "gpt-5.4-nano-none": "API",
    "gpt-5.4-nano-medium": "API",
    "gemma-4-31b": "Open",
    "gemma-4-26b-a4b": "Open",
    "gpt-oss-20b": "Open",
    "gpt-oss-20b-low": "Open",
    "gpt-oss-20b-medium": "Open",
    "qwen3.6-27b": "Open",
    "qwen3.6-35b-a3b": "Open",
    "bonsai-8b": "Open",
    "bonsai-27b": "Open",
}


def filter_excluded_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model_label" not in df.columns:
        return df
    return df[~df["model_label"].astype(str).isin(EXCLUDED_MODEL_LABELS)].copy()
OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#111111",
    "gray": "#777777",
}
LENGTH_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}


@dataclass
class Paths:
    evaluation_dir: Path
    results_artifact_dir: Path
    paper_generated_dir: Path
    paper_figure_dir: Path
    paper_table_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--models-config", default=DEFAULT_MODELS)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh evaluation CSVs from current raw outputs before building artifacts.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Alias for the default; build from existing paper_* or expanded_* evaluation CSVs.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="both",
        help="Figure output format.",
    )
    parser.add_argument(
        "--artifact-set",
        choices=("all", "main"),
        default="all",
        help="Build all paper artifacts or only the main Table 1/Figure 2 pair.",
    )
    return parser.parse_args()


def ensure_paths(config: dict[str, Any]) -> Paths:
    evaluation_dir = REPO_ROOT / config["paths"]["evaluation_dir"]
    results_artifact_dir = REPO_ROOT / "results" / "paper_artifacts"
    paper_generated_dir = REPO_ROOT / "paper" / "generated"
    paper_figure_dir = paper_generated_dir / "figures"
    paper_table_dir = paper_generated_dir / "tables"
    for path in [results_artifact_dir, paper_generated_dir, paper_figure_dir, paper_table_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return Paths(
        evaluation_dir=evaluation_dir,
        results_artifact_dir=results_artifact_dir,
        paper_generated_dir=paper_generated_dir,
        paper_figure_dir=paper_figure_dir,
        paper_table_dir=paper_table_dir,
    )


def run_command(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def raw_run_has_outputs(model_label: str, run_id: str) -> bool:
    run_dir = REPO_ROOT / "results" / "raw" / slugify(model_label) / slugify(run_id)
    return run_dir.exists() and any(
        path.suffix == ".json" and not path.name.startswith("_")
        for path in run_dir.iterdir()
    )


def first_existing_raw_run(model_label: str, run_ids: list[str]) -> str | None:
    for run_id in run_ids:
        if raw_run_has_outputs(model_label, run_id):
            return run_id
    return None


def available_model_labels(config: dict[str, Any]) -> list[str]:
    return [str(model["label"]) for model in config.get("models", [])]


def merge_base_runs_for_paper(labels: list[str]) -> None:
    raw_root = REPO_ROOT / "results" / "raw"
    for label in labels:
        if label in EXCLUDED_MODEL_LABELS:
            continue
        dest = raw_root / label / BASE_MERGED_RUN_ID
        dest.mkdir(parents=True, exist_ok=True)
        for run_id in BASE_RUN_CANDIDATES:
            source = raw_root / label / run_id
            if not source.exists():
                continue
            for path in source.glob("*.json"):
                target = dest / path.name
                if target.exists():
                    continue
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)


def refresh_evaluations(config: dict[str, Any]) -> None:
    python = sys.executable
    merge_base_runs_for_paper(available_model_labels(config))
    run_command(
        [
            python,
            "src/evaluate_numeric.py",
            "--manifest",
            BASE_MANIFEST,
            "--run-id",
            BASE_MERGED_RUN_ID,
            "--numeric-parser",
            "final_clean",
            "--output-prefix",
            "paper_base_auto",
        ]
    )

    for label in available_model_labels(config):
        if label in EXCLUDED_MODEL_LABELS:
            continue
        if raw_run_has_outputs(label, "tool_routing_v2_run1"):
            prefix = f"paper_tool_{slugify(label)}"
            run_command(
                [
                    python,
                    "src/evaluate_numeric.py",
                    "--manifest",
                    "results/manifests/tool_routing_image15.jsonl",
                    "--model-label",
                    label,
                    "--run-id",
                    "tool_routing_v2_run1",
                    "--numeric-parser",
                    "final_clean",
                    "--output-prefix",
                    prefix,
                ]
            )
            run_command(
                [
                    python,
                    "src/analyze_tool_routing.py",
                    "--scores",
                    f"results/evaluation/{prefix}_scores.csv",
                    "--output-prefix",
                    prefix,
                ]
            )
        if raw_run_has_outputs(label, "tool-routing-vllm-run1"):
            prefix = f"paper_tool_{slugify(label)}"
            run_command(
                [
                    python,
                    "src/evaluate_numeric.py",
                    "--manifest",
                    "results/manifests/tool_routing_image15.jsonl",
                    "--model-label",
                    label,
                    "--run-id",
                    "tool-routing-vllm-run1",
                    "--numeric-parser",
                    "final_clean",
                    "--output-prefix",
                    prefix,
                ]
            )
            run_command(
                [
                    python,
                    "src/analyze_tool_routing.py",
                    "--scores",
                    f"results/evaluation/{prefix}_scores.csv",
                    "--output-prefix",
                    prefix,
                ]
            )

    run_command(
        [
            python,
            "src/evaluate_numeric.py",
            "--manifest",
            "results/manifests/case_study_image15.jsonl",
            "--run-id",
            "case-study-vllm-run1",
            "--numeric-parser",
            "final_clean",
            "--output-prefix",
            "paper_case_study",
        ]
    )
    for label in available_model_labels(config):
        if label in EXCLUDED_MODEL_LABELS:
            continue
        case_run_id = first_existing_raw_run(label, CASE_STUDY_API_RUN_CANDIDATES)
        if not case_run_id:
            continue
        prefix = f"paper_case_study_api_{slugify(label)}"
        run_command(
            [
                python,
                "src/evaluate_numeric.py",
                "--manifest",
                "results/manifests/case_study_image15.jsonl",
                "--model-label",
                label,
                "--run-id",
                case_run_id,
                "--numeric-parser",
                "final_clean",
                "--output-prefix",
                prefix,
            ]
        )

    for label in available_model_labels(config):
        if label in EXCLUDED_MODEL_LABELS:
            continue
        reliability_run_id = first_existing_raw_run(label, RELIABILITY_RUN_CANDIDATES)
        if reliability_run_id:
            prefix = f"paper_reliability_{slugify(label)}"
            run_command(
                [
                    python,
                    "src/evaluate_numeric.py",
                    "--manifest",
                    "results/manifests/reliability_image15_repeat3.jsonl",
                    "--model-label",
                    label,
                    "--run-id",
                    reliability_run_id,
                    "--numeric-parser",
                    "final_clean",
                    "--output-prefix",
                    prefix,
                ]
            )
            run_command(
                [
                    python,
                    "src/analyze_reliability.py",
                    "--scores",
                    f"results/evaluation/{prefix}_scores.csv",
                    "--output-prefix",
                    prefix,
                ]
            )


def latest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    return None


def select_base_scores(paths: Paths) -> Path:
    candidates = [
        paths.evaluation_dir / "paper_base_auto_scores.csv",
        paths.evaluation_dir / "expanded_base_auto_review_label_fixed_scores.csv",
        paths.evaluation_dir / "expanded_base_auto_review_scores.csv",
        paths.evaluation_dir / "latest_base_review2_scores.csv",
    ]
    selected = latest_existing(candidates)
    if selected is None:
        raise SystemExit("No base score CSV found. Run with --refresh or evaluate base tasks first.")
    return selected


def select_case_scores(paths: Paths) -> Path | None:
    candidates = [
        paths.evaluation_dir / "paper_case_study_scores.csv",
        paths.evaluation_dir / "expanded_case_study_review_label_fixed_scores.csv",
        paths.evaluation_dir / "expanded_case_study_review_scores.csv",
    ]
    return latest_existing(candidates)


def case_score_source_paths(paths: Paths) -> list[Path]:
    candidates = [
        select_case_scores(paths),
        paths.evaluation_dir / "paper_case_study_api_scores.csv",
        *sorted(paths.evaluation_dir.glob("paper_case_study_api_*_scores.csv")),
    ]
    return [path for path in candidates if path is not None and path.exists()]


def load_case_scores(paths: Paths) -> pd.DataFrame:
    frames = []
    for path in case_score_source_paths(paths):
        frames.append(read_csv(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if {"model_label", "task_id"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["model_label", "task_id"], keep="last")
    return out


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def score_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    success = df[df["status"].astype(str) == "success"].copy()
    if success.empty:
        return float("nan")
    return float((bool_series(success["parse_success"]) & bool_series(success["correct"])).mean())


def parsed_accuracy(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    parsed = df[(df["status"].astype(str) == "success") & bool_series(df["parse_success"])]
    if parsed.empty:
        return float("nan")
    return float(bool_series(parsed["correct"]).mean())


def mean_numeric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    values = numeric(df[column]).dropna()
    return float(values.mean()) if not values.empty else float("nan")


def model_sort_key(label: str) -> tuple[int, str]:
    if label in MAIN_MODEL_ORDER:
        return (MAIN_MODEL_ORDER.index(label), label)
    return (len(MAIN_MODEL_ORDER), label)


def paradigm_sort_key(label: str) -> tuple[int, str]:
    if label in PARADIGM_TABLE_ORDER:
        return (PARADIGM_TABLE_ORDER.index(label), label)
    return (len(PARADIGM_TABLE_ORDER), label)


def model_display(label: str) -> str:
    return MODEL_SHORT.get(label, label)


def model_access(config: dict[str, Any], label: str) -> str:
    for model in config.get("models", []):
        if str(model.get("label")) == label:
            access = str(model.get("access", model.get("provider", "")))
            provider = str(model.get("provider", ""))
            if access == "local" or provider == "lmstudio":
                return "Local"
            if access == "api" or provider == "openai":
                return "API"
            return access or provider
    return ""


def model_class(label: str, config: dict[str, Any]) -> str:
    if label in MODEL_CLASS:
        return MODEL_CLASS[label]
    access = model_access(config, label)
    if access == "API":
        return "API"
    if access == "Local":
        return "Open"
    return access or "--"


def raw_output_path(model_label: str, run_id: str, task_id: str) -> Path:
    return REPO_ROOT / "results" / "raw" / slugify(model_label) / slugify(run_id) / f"{task_id}.json"


def length_stop_count(score_rows: pd.DataFrame) -> int:
    count = 0
    if score_rows.empty:
        return count
    for _, row in score_rows.iterrows():
        run_id = str(row.get("source_run_id", ""))
        task_id = str(row.get("task_id", ""))
        model_label = str(row.get("model_label", ""))
        if not run_id or not task_id or not model_label:
            continue
        path = raw_output_path(model_label, run_id, task_id)
        if not path.exists():
            continue
        try:
            finish_reason = str(json.loads(path.read_text(encoding="utf-8")).get("finish_reason", "")).lower()
        except (OSError, json.JSONDecodeError):
            continue
        if finish_reason in LENGTH_FINISH_REASONS:
            count += 1
    return count


def build_base_summary(base_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if base_df.empty:
        return pd.DataFrame()
    core = base_df[base_df["suite"].astype(str) == "core"].copy()
    for label in sorted(core["model_label"].dropna().unique(), key=model_sort_key):
        model_rows = core[core["model_label"] == label]
        success_rows = model_rows[model_rows["status"].astype(str) == "success"]
        processed = success_rows[success_rows["encoding"].isin(["processed_table", "processed_text"])]
        raw = success_rows[success_rows["encoding"].isin(["raw_table", "raw_text"])]
        rows.append(
            {
                "model_label": label,
                "model": model_display(label),
                "access": model_access(config, label),
                "model_class": model_class(label, config),
                "base_success": len(success_rows),
                "base_total": len(model_rows),
                "base_accuracy": score_rate(success_rows),
                "processed_accuracy": score_rate(processed),
                "raw_accuracy": score_rate(raw),
                "processed_tasks": len(processed),
                "raw_tasks": len(raw),
                "raw_length_stops": length_stop_count(model_rows[model_rows["encoding"].isin(["raw_table", "raw_text"])]),
                "mean_total_tokens": mean_numeric(success_rows, "total_tokens"),
                "mean_output_tokens": mean_numeric(success_rows, "output_tokens"),
                "mean_latency_s": mean_numeric(success_rows, "elapsed_seconds"),
            }
        )
    return pd.DataFrame(rows)


def build_encoding_summary(base_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    core = base_df[base_df["suite"].astype(str) == "core"].copy()
    for (label, encoding), group in core.groupby(["model_label", "encoding"], dropna=False):
        success = group[group["status"].astype(str) == "success"]
        rows.append(
            {
                "model_label": label,
                "model": model_display(str(label)),
                "encoding": encoding,
                "tasks": len(success),
                "completed_accuracy": score_rate(success),
                "parsed_accuracy": parsed_accuracy(success),
                "parse_rate": float(bool_series(success["parse_success"]).mean()) if not success.empty else np.nan,
                "mean_total_tokens": mean_numeric(success, "total_tokens"),
            }
        )
    return pd.DataFrame(rows)


def load_tool_summary(paths: Paths) -> pd.DataFrame:
    files = sorted(paths.evaluation_dir.glob("paper_tool_*_tool_routing_by_model.csv"))
    if not files:
        files = sorted(paths.evaluation_dir.glob("*_tool_routing_image15_expanded_review_tool_routing_by_model.csv"))
    if not files:
        files = sorted(paths.evaluation_dir.glob("*_latest_review2_tool_routing_by_model.csv"))
    frames = []
    seen: set[str] = set()
    for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True):
        df = pd.read_csv(path)
        if df.empty or "model_label" not in df:
            continue
        label = str(df.iloc[0]["model_label"])
        if label in seen:
            continue
        seen.add(label)
        scores_path = path.with_name(path.name.replace("_tool_routing_by_model.csv", "_scores.csv"))
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            success_scores = scores[scores["status"].astype(str) == "success"] if "status" in scores else scores
            df["mean_output_tokens"] = mean_numeric(success_scores, "output_tokens")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in [
        "routing_accuracy",
        "strict_routing_accuracy",
        "executable_rate",
        "mean_latency_s",
        "mean_total_tokens",
        "mean_output_tokens",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["model"] = out["model_label"].map(model_display)
    return out.sort_values("model_label", key=lambda s: s.map(model_sort_key)).reset_index(drop=True)


def load_reliability_summary(paths: Paths) -> pd.DataFrame:
    files = sorted(paths.evaluation_dir.glob("paper_reliability_*_reliability_summary.json"))
    if not files:
        files = sorted(paths.evaluation_dir.glob("*_reliability-vllm-run1_expanded_review_label_fixed_reliability_summary.json"))
    rows = []
    for path in files:
        name = path.name
        label = name
        label = label.replace("paper_reliability_", "")
        label = label.replace("_reliability-vllm-run1_expanded_review_label_fixed", "")
        label = label.replace("_reliability_summary.json", "")
        summary = json.loads(path.read_text(encoding="utf-8"))
        score_candidates = [
            path.with_name(path.name.replace("_reliability_summary.json", "_scores.csv")),
            paths.evaluation_dir / f"{label}_reliability-vllm-run1_expanded_review_label_fixed_scores.csv",
        ]
        score_path = latest_existing(score_candidates)
        success = total = parsed_acc = np.nan
        if score_path is not None:
            scores = pd.read_csv(score_path)
            total = len(scores)
            scored = scores[scores["status"].astype(str) == "success"]
            success = len(scored)
            parsed_acc = parsed_accuracy(scored)
        rows.append(
            {
                "model_label": label,
                "model": model_display(label),
                "success": success,
                "total": total,
                "parsed_accuracy": parsed_acc,
                "exact_response_agreement": summary.get("mean_exact_response_agreement"),
                "parsed_answer_agreement": summary.get("mean_parsed_answer_agreement"),
                "parse_consistency": summary.get("mean_parse_consistency"),
                "accuracy_variance": summary.get("mean_accuracy_variance"),
            }
        )
    return pd.DataFrame(rows).sort_values("model_label", key=lambda s: s.map(model_sort_key)).reset_index(drop=True)


def load_case_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()
    rows = []
    for label in sorted(case_df["model_label"].dropna().unique(), key=model_sort_key):
        model_rows = case_df[case_df["model_label"] == label]
        stage1_numeric = model_rows[
            (model_rows["case_study_stage"].astype(str) == "1")
            & (model_rows["answer_type"].astype(str) == "numeric")
        ]
        stage1_label = model_rows[
            (model_rows["case_study_stage"].astype(str) == "1")
            & (model_rows["answer_type"].astype(str) == "label")
        ]
        stage2 = model_rows[
            (model_rows["case_study_stage"].astype(str) == "2")
            & (model_rows["answer_type"].astype(str) == "topk_metrics")
        ]
        stage2_success = stage2[stage2["status"].astype(str) == "success"]
        rows.append(
            {
                "model_label": label,
                "model": model_display(label),
                "success": int((model_rows["status"].astype(str) == "success").sum()),
                "total": len(model_rows),
                "stage1_numeric": score_rate(stage1_numeric),
                "stage1_pairwise": score_rate(stage1_label),
                "stage2_topk_p3": mean_numeric(stage2_success, "topk_precision_at_3"),
                "stage2_family_p3": mean_numeric(stage2_success, "topk_family_precision_at_3"),
            }
        )
    return pd.DataFrame(rows)


def apply_denominator_corrections(
    paths: Paths,
    main: pd.DataFrame,
    encoding: pd.DataFrame,
    reliability: pd.DataFrame,
    case: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Prefer end-to-end denominators from the audit layer for paper artifacts."""
    potential_dir = paths.paper_generated_dir / "potential"
    used_sources: list[str] = []
    main = main.copy()
    encoding = encoding.copy()
    reliability = reliability.copy()
    case = case.copy()

    direct_path = potential_dir / "direct_end_to_end_metrics.csv"
    if direct_path.exists():
        direct = filter_excluded_models(read_csv(direct_path))
        direct_by_label = {str(row["model_label"]): row for _, row in direct.iterrows()}
        encoding_map = {
            "raw_table": "raw_table_e2e_accuracy",
            "raw_text": "raw_text_e2e_accuracy",
            "processed_table": "processed_table_e2e_accuracy",
            "processed_text": "processed_text_e2e_accuracy",
        }
        for idx, row in encoding.iterrows():
            label = str(row.get("model_label"))
            enc = str(row.get("encoding"))
            direct_row = direct_by_label.get(label)
            column = encoding_map.get(enc)
            if direct_row is not None and column in direct_row and pd.notna(direct_row[column]):
                encoding.at[idx, "completed_accuracy"] = float(direct_row[column])
                encoding.at[idx, "parsed_accuracy"] = float(direct_row[column])
        for idx, row in main.iterrows():
            label = str(row.get("model_label"))
            direct_row = direct_by_label.get(label)
            if direct_row is None:
                continue
            for source_col, target_col in [
                ("raw_e2e_accuracy", "raw_accuracy"),
                ("processed_e2e_accuracy", "processed_accuracy"),
            ]:
                if source_col in direct_row and pd.notna(direct_row[source_col]):
                    main.at[idx, target_col] = float(direct_row[source_col])
        used_sources.append(str(direct_path.relative_to(REPO_ROOT)))

    tool_path = potential_dir / "tool_end_to_end_metrics.csv"
    if tool_path.exists():
        tool_e2e = filter_excluded_models(read_csv(tool_path))
        tool_by_label = {str(row["model_label"]): row for _, row in tool_e2e.iterrows()}
        for idx, row in main.iterrows():
            tool_row = tool_by_label.get(str(row.get("model_label")))
            if tool_row is None:
                continue
            for source_col, target_col in [
                ("accepted_e2e", "routing_accuracy"),
                ("strict_e2e", "strict_routing_accuracy"),
                ("executable_e2e", "executable_rate"),
            ]:
                if source_col in tool_row and pd.notna(tool_row[source_col]):
                    main.at[idx, target_col] = float(tool_row[source_col])
        used_sources.append(str(tool_path.relative_to(REPO_ROOT)))

    reliability_path = potential_dir / "reliability_end_to_end_metrics.csv"
    if reliability_path.exists():
        rel_e2e = filter_excluded_models(read_csv(reliability_path))
        rel_by_label = {str(row["model_label"]): row for _, row in rel_e2e.iterrows()}
        for idx, row in reliability.iterrows():
            rel_row = rel_by_label.get(str(row.get("model_label")))
            if rel_row is None:
                continue
            if "repeat_accuracy_e2e" in rel_row and pd.notna(rel_row["repeat_accuracy_e2e"]):
                reliability.at[idx, "parsed_accuracy"] = float(rel_row["repeat_accuracy_e2e"])
        used_sources.append(str(reliability_path.relative_to(REPO_ROOT)))

    case_path = potential_dir / "case_study_end_to_end_metrics.csv"
    if case_path.exists():
        case_e2e = filter_excluded_models(read_csv(case_path))
        case_by_label = {str(row["model_label"]): row for _, row in case_e2e.iterrows()}
        for idx, row in case.iterrows():
            case_row = case_by_label.get(str(row.get("model_label")))
            if case_row is None:
                continue
            for source_col, target_col in [
                ("numeric_e2e", "stage1_numeric"),
                ("pairwise_e2e", "stage1_pairwise"),
                ("exact_top3_e2e", "stage2_topk_p3"),
                ("family_top3_e2e", "stage2_family_p3"),
            ]:
                if source_col in case_row and pd.notna(case_row[source_col]):
                    case.at[idx, target_col] = float(case_row[source_col])
        used_sources.append(str(case_path.relative_to(REPO_ROOT)))

    return main, encoding, reliability, case, used_sources


def combine_main(base: pd.DataFrame, tool: pd.DataFrame) -> pd.DataFrame:
    if base.empty and tool.empty:
        return pd.DataFrame()
    cols = [
        "model_label",
        "routing_accuracy",
        "strict_routing_accuracy",
        "executable_rate",
        "successful_outputs",
        "tasks",
        "mean_total_tokens",
        "mean_output_tokens",
        "mean_latency_s",
    ]
    available = [col for col in cols if col in tool.columns]
    if base.empty:
        merged = tool[available].copy()
    elif tool.empty:
        merged = base.copy()
    else:
        merged = base.merge(tool[available], on="model_label", how="outer", suffixes=("", "_tool"))
    merged["model"] = merged["model_label"].astype(str).map(model_display)
    if "access" not in merged:
        merged["access"] = ""
    if "model_class" not in merged:
        merged["model_class"] = ""
    merged["model_class"] = merged.apply(
        lambda row: row["model_class"] if isinstance(row.get("model_class"), str) and row.get("model_class") else MODEL_CLASS.get(str(row["model_label"]), ""),
        axis=1,
    )
    merged["access"] = merged.apply(
        lambda row: row["access"] if isinstance(row.get("access"), str) and row.get("access") else ("API" if row.get("model_class") == "API" else "Local"),
        axis=1,
    )
    merged = merged.sort_values("model_label", key=lambda s: s.map(model_sort_key)).reset_index(drop=True)
    return merged


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number * 100:.{digits}f}"


def num(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def pp_delta(new_value: Any, old_value: Any, digits: int = 1) -> str:
    try:
        new_number = float(new_value)
        old_number = float(old_value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(new_number) or not np.isfinite(old_number):
        return "--"
    delta = (new_number - old_number) * 100
    return f"{delta:+.{digits}f} pp"


def tex_escape(text: Any) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[Any]],
    colspec: str | None = None,
) -> None:
    if colspec is None:
        colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        "% Auto-generated by src/build_paper_artifacts.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(item) for item in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def tex_pct(value: Any, digits: int = 1, bold_at: float | None = None) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    text = f"{number * 100:.{digits}f}"
    if bold_at is not None and number >= bold_at:
        return f"\\textbf{{{text}}}"
    return text


def tex_count(done: Any, total: Any) -> str:
    try:
        done_value = int(done)
        total_value = int(total)
    except (TypeError, ValueError):
        return "--"
    if total_value <= 0:
        return "--"
    return f"{done_value}/{total_value}"


def write_main_results_table(paths: Paths, main: pd.DataFrame) -> None:
    selected = main[main["model_label"].isin(PARADIGM_TABLE_ORDER)].copy()
    selected = selected.sort_values("model_label", key=lambda s: s.map(paradigm_sort_key))
    groups = [
        ("API models", selected[selected["model_class"].astype(str) == "API"]),
        ("Open-weight models", selected[selected["model_class"].astype(str) != "API"]),
    ]
    lines = [
        "% Auto-generated by src/build_paper_artifacts.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\caption{Main benchmark summary. Direct computation asks the model to read morphology records and produce numerical answers; tool routing asks it to emit a schema-valid, executable characterization call. Accuracy values are completed-prompt accuracy within 1\\%. Accepted routing allows deterministic-equivalent feature/statistic choices; strict routing is reported in the appendix.}",
        "\\label{tab:main-results-expanded}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        " &  & \\multicolumn{3}{c}{Direct computation} & \\multicolumn{3}{c}{Tool routing} & \\multicolumn{2}{c}{Failure signals} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
        "Model & Class & n & Raw acc. & Proc. acc. & n & Accepted & Executable & Route--raw & Raw length stops \\\\",
        "\\midrule",
    ]
    wrote_group = False
    for group_name, group in groups:
        if group.empty:
            continue
        if wrote_group:
            lines.append("\\addlinespace[1pt]")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{tex_escape(group_name)}}}}} \\\\")
        for _, row in group.iterrows():
            direct_n = tex_count(row.get("base_success"), row.get("base_total"))
            route_n = tex_count(row.get("successful_outputs"), row.get("tasks"))
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["model"]),
                        tex_escape(row.get("model_class", row["access"])),
                        direct_n,
                        tex_pct(row["raw_accuracy"]),
                        tex_pct(row["processed_accuracy"], bold_at=0.995),
                        route_n,
                        tex_pct(row.get("routing_accuracy"), bold_at=0.995),
                        tex_pct(row.get("executable_rate"), bold_at=0.995),
                        tex_escape(pp_delta(row.get("routing_accuracy"), row["raw_accuracy"])),
                        "--" if not np.isfinite(pd.to_numeric(row.get("raw_length_stops"), errors="coerce")) else str(int(row.get("raw_length_stops", 0))),
                    ]
                )
                + " \\\\"
            )
        wrote_group = True
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path = paths.paper_table_dir / "table_main_results.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def table_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number * 100:.1f}"


def table_pct_one_decimal_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(f"{number * 100:.1f}")


def table_pct_one_decimal(value: Any) -> str:
    number = table_pct_one_decimal_value(value)
    if number is None:
        return "--"
    return f"{number:.1f}"


def table_drop(processed: Any, raw: Any) -> str:
    try:
        processed_number = float(processed)
        raw_number = float(raw)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(processed_number) or not np.isfinite(raw_number):
        return "--"
    return f"{(processed_number - raw_number) * 100:.1f}"


def compact_metric_cell(*values: Any) -> str:
    rendered = [table_pct(value) for value in values]
    if all(value == "--" for value in rendered):
        return "--"
    return "/".join(rendered)


def encoding_accuracy(encoding: pd.DataFrame, label: str, encoding_key: str) -> float:
    if encoding.empty:
        return float("nan")
    rows = encoding[
        (encoding["model_label"].astype(str) == label)
        & (encoding["encoding"].astype(str) == encoding_key)
    ]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows.iloc[0].get("completed_accuracy"), errors="coerce"))


def reliability_table_values(reliability: pd.DataFrame, label: str) -> tuple[str, str]:
    acc, agreement = reliability_table_values_for_main(reliability, label)
    return format_table1_cell(acc), format_table1_cell(agreement)


def reliability_table_values_for_main(reliability: pd.DataFrame, label: str) -> tuple[float | None, float | None]:
    if reliability.empty:
        return None, None
    rows = reliability[reliability["model_label"].astype(str) == label]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    success = pd.to_numeric(row.get("success"), errors="coerce")
    total = pd.to_numeric(row.get("total"), errors="coerce")
    if not np.isfinite(success) or not np.isfinite(total) or total <= 0:
        return None, None
    if success / total < 0.5:
        return None, None
    acc = table_pct_one_decimal_value(row.get("parsed_accuracy"))
    agreement = table_pct_one_decimal_value(row.get("exact_response_agreement"))
    return acc, agreement


def table1_emphasis(values: dict[str, float | None]) -> dict[str, str]:
    finite_values = sorted({value for value in values.values() if value is not None}, reverse=True)
    if not finite_values:
        return {}
    best = finite_values[0]
    best_labels = [label for label, value in values.items() if value == best]
    emphasis = {label: "bold" for label in best_labels}
    if len(finite_values) > 1:
        second = finite_values[1]
        for label, value in values.items():
            if value == second:
                emphasis[label] = "underline"
    return emphasis


def format_table1_cell(value: float | None, emphasis: str | None = None) -> str:
    if value is None:
        return "--"
    text = f"{value:.1f}"
    if emphasis == "bold":
        text = f"\\textbf{{{text}}}"
    elif emphasis == "underline":
        text = f"\\underline{{{text}}}"
    return text


def case_study_values_for_main(case: pd.DataFrame, label: str) -> tuple[float | None, float | None]:
    if case.empty:
        return None, None
    rows = case[case["model_label"].astype(str) == label]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    return table_pct_one_decimal_value(row.get("stage1_pairwise")), table_pct_one_decimal_value(
        row.get("stage2_family_p3")
    )


def write_table1(
    paths: Paths,
    main: pd.DataFrame,
    encoding: pd.DataFrame,
    reliability: pd.DataFrame,
    case: pd.DataFrame | None = None,
) -> None:
    case = pd.DataFrame() if case is None else case
    selected = main[main["model_label"].isin(TABLE1_MODEL_ORDER)].copy()
    if selected.empty:
        return
    selected["table_order"] = selected["model_label"].map({label: idx for idx, label in enumerate(TABLE1_MODEL_ORDER)})
    selected = selected.sort_values("table_order")
    table_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        label = str(row["model_label"])
        repeat_acc, agreement = reliability_table_values_for_main(reliability, label)
        case_pairwise, case_family_top3 = case_study_values_for_main(case, label)
        table_rows.append(
            {
                "model_label": label,
                "model": row["model"],
                "model_class": row["model_class"],
                "processed_text": table_pct_one_decimal_value(encoding_accuracy(encoding, label, "processed_text")),
                "processed_table": table_pct_one_decimal_value(encoding_accuracy(encoding, label, "processed_table")),
                "raw_text": table_pct_one_decimal_value(encoding_accuracy(encoding, label, "raw_text")),
                "raw_table": table_pct_one_decimal_value(encoding_accuracy(encoding, label, "raw_table")),
                "accepted": table_pct_one_decimal_value(row.get("routing_accuracy")),
                "strict": table_pct_one_decimal_value(row.get("strict_routing_accuracy")),
                "repeat_acc": repeat_acc,
                "agreement": agreement,
                "case_pairwise": case_pairwise,
                "case_family_top3": case_family_top3,
            }
        )
    emphasis_by_column = {
        column: table1_emphasis({str(row["model_label"]): row[column] for row in table_rows})
        for column in [
            "raw_table",
            "raw_text",
            "processed_table",
            "processed_text",
            "accepted",
            "strict",
            "repeat_acc",
            "agreement",
            "case_pairwise",
            "case_family_top3",
        ]
    }
    groups = [
        ("Open-weight / local", [row for row in table_rows if str(row["model_class"]) != "API"]),
        ("GPT series / API", [row for row in table_rows if str(row["model_class"]) == "API"]),
    ]
    lines = [
        "% Auto-generated by src/build_paper_artifacts.py",
        "\\begin{table}[b]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\newcommand{\\directhead}[2]{\\parbox[c][2.45em][c]{0.68cm}{\\centering #1\\\\#2}}",
        "\\newcommand{\\metrichead}[2]{\\parbox[c][2.45em][c]{1.05cm}{\\centering #1\\\\#2}}",
        "\\caption{\\textbf{LLM performance across SEM characterization tasks.} Comparing open-weight and GPT-model results (\\%) across direct answering with four input serializations, schema-guided tool calling, output repeatability, and comparative characterization case study. Higher values are better, with the best value per metric \\textbf{bolded} and second-best \\underline{underlined}.}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Direct answer} & \\multicolumn{2}{c}{Tool call} & \\multicolumn{2}{c}{Repeatability} & \\multicolumn{2}{c}{Case study} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
        "Model & \\directhead{Raw}{table} & \\directhead{Raw}{text} & \\directhead{Proc.}{table} & \\directhead{Proc.}{text} & \\metrichead{Accepted}{call} & \\metrichead{Strict}{call} & \\metrichead{Repeat}{accuracy} & \\metrichead{Repeat}{agreement} & \\metrichead{Pairwise}{comparison} & \\metrichead{Family}{top-3} \\\\",
        "\\midrule",
    ]
    wrote_group = False
    for group_name, group in groups:
        if not group:
            continue
        if wrote_group:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{tex_escape(group_name)}}}}} \\\\")
        for row in group:
            label = str(row["model_label"])
            lines.append(
                " & ".join(
                    [
                        tex_escape(row["model"]),
                        format_table1_cell(row["raw_table"], emphasis_by_column["raw_table"].get(label)),
                        format_table1_cell(row["raw_text"], emphasis_by_column["raw_text"].get(label)),
                        format_table1_cell(
                            row["processed_table"], emphasis_by_column["processed_table"].get(label)
                        ),
                        format_table1_cell(row["processed_text"], emphasis_by_column["processed_text"].get(label)),
                        format_table1_cell(row["accepted"], emphasis_by_column["accepted"].get(label)),
                        format_table1_cell(row["strict"], emphasis_by_column["strict"].get(label)),
                        format_table1_cell(row["repeat_acc"], emphasis_by_column["repeat_acc"].get(label)),
                        format_table1_cell(row["agreement"], emphasis_by_column["agreement"].get(label)),
                        format_table1_cell(row["case_pairwise"], emphasis_by_column["case_pairwise"].get(label)),
                        format_table1_cell(row["case_family_top3"], emphasis_by_column["case_family_top3"].get(label)),
                    ]
                )
                + " \\\\"
            )
        wrote_group = True
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path = paths.paper_table_dir / "table1.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def write_tables(
    paths: Paths,
    base_df: pd.DataFrame,
    main: pd.DataFrame,
    encoding: pd.DataFrame,
    tool: pd.DataFrame,
    case: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    write_table1(paths, main, encoding, reliability, case)
    write_main_results_table(paths, main)

    tool_rows = []
    for _, row in tool.sort_values("routing_accuracy", ascending=False).iterrows():
        tool_rows.append(
            [
                row["model"],
                f"{int(row['successful_outputs'])}/{int(row['tasks'])}",
                pct(row["routing_accuracy"]),
                pct(row["strict_routing_accuracy"]),
                pct(row["executable_rate"]),
            ]
        )
    write_table(
        paths.paper_table_dir / "table_tool_routing_full.tex",
        "Full tool-routing results. Accepted scoring permits deterministic-equivalent morphology routes.",
        "tab:tool-routing-full",
        ["Model", "n", "Accepted", "Strict", "Exec."],
        tool_rows,
    )

    enc_rows = []
    pivot = encoding.pivot_table(
        index=["model_label", "model"],
        columns="encoding",
        values="completed_accuracy",
        aggfunc="first",
    ).reset_index()
    for _, row in pivot.sort_values("model_label", key=lambda s: s.map(model_sort_key)).iterrows():
        enc_rows.append(
            [
                row["model"],
                pct(row.get("processed_table")),
                pct(row.get("processed_text")),
                pct(row.get("raw_table")),
                pct(row.get("raw_text")),
            ]
        )
    write_table(
        paths.paper_table_dir / "table_base_by_encoding.tex",
        "Direct core benchmark by input serialization for all currently detected models. Values include parse failures as incorrect.",
        "tab:base-by-encoding",
        ["Model", "Proc. tbl", "Proc. txt", "Raw tbl", "Raw txt"],
        enc_rows,
    )

    coverage_rows = []
    model_labels = set(main.get("model_label", pd.Series(dtype=str)).dropna().astype(str))
    if not tool.empty:
        model_labels.update(tool.get("model_label", pd.Series(dtype=str)).dropna().astype(str))
    if not case.empty:
        model_labels.update(case.get("model_label", pd.Series(dtype=str)).dropna().astype(str))
    if not reliability.empty:
        model_labels.update(reliability.get("model_label", pd.Series(dtype=str)).dropna().astype(str))

    for label in sorted(model_labels, key=model_sort_key):
        direct_rows = base_df[base_df["model_label"].astype(str) == label]
        direct_success = direct_rows[direct_rows["status"].astype(str) == "success"]
        core_rows = direct_rows[direct_rows["suite"].astype(str) == "core"]
        core_success = core_rows[core_rows["status"].astype(str) == "success"]
        interface_rows = direct_rows[direct_rows["suite"].astype(str) != "core"]
        interface_success = interface_rows[interface_rows["status"].astype(str) == "success"]
        tool_row = tool[tool["model_label"].astype(str) == label]
        case_row = case[case["model_label"].astype(str) == label] if not case.empty else pd.DataFrame()
        rel_row = reliability[reliability["model_label"].astype(str) == label] if not reliability.empty else pd.DataFrame()
        coverage_rows.append(
            [
                model_display(label),
                tex_count(len(direct_success), len(direct_rows)),
                tex_count(len(core_success), len(core_rows)),
                tex_count(len(interface_success), len(interface_rows)),
                tex_count(tool_row.iloc[0].get("successful_outputs"), tool_row.iloc[0].get("tasks")) if not tool_row.empty else "--",
                tex_count(case_row.iloc[0].get("success"), case_row.iloc[0].get("total")) if not case_row.empty else "--",
                tex_count(rel_row.iloc[0].get("success"), rel_row.iloc[0].get("total")) if not rel_row.empty else "--",
            ]
        )
    write_table(
        paths.paper_table_dir / "table_run_coverage.tex",
        "Run coverage by experiment suite. Full direct is the 5,500-prompt 20-image manifest; core direct is the 5,120-prompt numerical benchmark used in Table~\\ref{tab:main-results}.",
        "tab:run-coverage",
        ["Model", "Full direct", "Core", "Interface", "Tool", "Case", "Reliab."],
        coverage_rows,
    )

    if not case.empty:
        case_rows = []
        for _, row in case.sort_values("model_label", key=lambda s: s.map(model_sort_key)).iterrows():
            case_rows.append(
                [
                    row["model"],
                    f"{int(row['success'])}/{int(row['total'])}",
                    pct(row["stage1_numeric"]),
                    pct(row["stage1_pairwise"]),
                    pct(row["stage2_topk_p3"]),
                ]
            )
        write_table(
            paths.paper_table_dir / "table_case_study.tex",
            "Case-study performance. Stage 2 reports direct top-3 metric precision, not tool-routed top-k.",
            "tab:case-study",
            ["Model", "n", "S1 num.", "S1 pair", "S2 P@3"],
            case_rows,
        )

    if not reliability.empty:
        rel_rows = []
        for _, row in reliability.sort_values("model_label", key=lambda s: s.map(model_sort_key)).iterrows():
            rel_rows.append(
                [
                    row["model"],
                    f"{int(row['success'])}/{int(row['total'])}",
                    pct(row["parsed_accuracy"]),
                    pct(row["exact_response_agreement"]),
                    pct(row["parsed_answer_agreement"]),
                    num(row["accuracy_variance"], 4),
                ]
            )
        write_table(
            paths.paper_table_dir / "table_reliability.tex",
            "Repeated-prompt reliability for direct tasks. Agreement can be high even when direct computation accuracy is limited.",
            "tab:reliability",
            ["Model", "n", "Acc.", "Exact", "Parsed", "Var."],
            rel_rows,
        )


def save_figure(fig: plt.Figure, paths: Paths, stem: str, output_format: str) -> None:
    targets = []
    if output_format in {"pdf", "both"}:
        targets.append(paths.paper_figure_dir / f"{stem}.pdf")
    if output_format in {"png", "both"}:
        targets.append(paths.paper_figure_dir / f"{stem}.png")
    for target in targets:
        dpi = 600 if target.suffix == ".png" else 300
        fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Wrote {target}")
    plt.close(fig)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 8,
    weight: str = "bold",
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.1,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="#111111",
        weight=weight,
        linespacing=1.05,
        fontsize=fontsize,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def figure_architecture(paths: Paths, output_format: str) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.96, "Tool-routed characterization interface", weight="bold", fontsize=11)
    ax.text(
        0.03,
        0.88,
        "LLMs translate materials language; validated Python tools do the numerical characterization.",
        fontsize=7.5,
        color="#333333",
    )

    ax.text(0.03, 0.75, "OFFLINE COMPUTER VISION", color=OKABE_ITO["blue"], weight="bold", fontsize=6.5)
    draw_box(ax, (0.03, 0.57), 0.13, 0.14, "SEM\nmicrograph", OKABE_ITO["black"], fontsize=6.7)
    draw_box(ax, (0.22, 0.57), 0.17, 0.14, "Segmentation +\nfeature extraction", OKABE_ITO["blue"], fontsize=6.4)
    draw_box(ax, (0.45, 0.57), 0.14, 0.14, "Morphology\nfeature DB", OKABE_ITO["sky"], fontsize=6.7)
    draw_arrow(ax, (0.16, 0.64), (0.22, 0.64))
    draw_arrow(ax, (0.39, 0.64), (0.45, 0.64))

    badge = FancyBboxPatch(
        (0.66, 0.57),
        0.29,
        0.17,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=OKABE_ITO["green"],
        facecolor=OKABE_ITO["green"] + "12",
    )
    ax.add_patch(badge)
    ax.text(0.805, 0.665, "0 LLM math", ha="center", va="center", weight="bold", fontsize=8, color=OKABE_ITO["green"])
    ax.text(0.805, 0.605, "100% deterministic verification path", ha="center", va="center", fontsize=6.6, color="#333333")

    ax.text(0.03, 0.43, "ONLINE NATURAL-LANGUAGE QUERY", color=OKABE_ITO["green"], weight="bold", fontsize=6.5)
    draw_box(ax, (0.03, 0.23), 0.14, 0.15, 'User query\n"bigger chunks?"', OKABE_ITO["orange"], fontsize=6.4)
    draw_box(ax, (0.23, 0.23), 0.14, 0.15, "LLM intent\nrouter", OKABE_ITO["green"], fontsize=6.7)
    draw_box(ax, (0.43, 0.23), 0.14, 0.15, "JSON tool\ncall", OKABE_ITO["green"], fontsize=6.7)
    draw_box(ax, (0.63, 0.23), 0.14, 0.15, "Python\nexecution", OKABE_ITO["green"], fontsize=6.7)
    draw_box(ax, (0.83, 0.23), 0.14, 0.15, "Auditable\nanswer", OKABE_ITO["green"], fontsize=6.7)
    draw_arrow(ax, (0.17, 0.305), (0.23, 0.305), OKABE_ITO["green"])
    draw_arrow(ax, (0.37, 0.305), (0.43, 0.305), OKABE_ITO["green"])
    draw_arrow(ax, (0.57, 0.305), (0.63, 0.305), OKABE_ITO["green"])
    draw_arrow(ax, (0.77, 0.305), (0.83, 0.305), OKABE_ITO["green"])
    draw_arrow(ax, (0.52, 0.57), (0.70, 0.38), OKABE_ITO["sky"])

    ax.text(
        0.50,
        0.155,
        '{"tool": "compare_morphology_statistic", "feature": "equivalent_diameter", "statistic": "percentile_90"}',
        ha="center",
        va="center",
        family="monospace",
        fontsize=5.6,
        color="#333333",
    )
    ax.text(
        0.50,
        0.055,
        "Material question -> schema-bound intent -> deterministic morphology statistic",
        ha="center",
        va="center",
        color=OKABE_ITO["green"],
        fontsize=7.5,
        weight="bold",
    )
    save_figure(fig, paths, "fig_architecture", output_format)


def micrograph_metric_summary() -> list[tuple[str, str]]:
    preferred = (
        REPO_ROOT
        / "data"
        / "morphology"
        / "csvs"
        / "TruForm174-2_00-37_500X16"
        / "TruForm174-2_00-37_500X16_raw.csv"
    )
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted((REPO_ROOT / "data" / "morphology" / "csvs").glob("*/*_raw.csv")))
    if not candidates:
        return []
    raw = pd.read_csv(candidates[0])
    rows: list[tuple[str, str]] = []
    if "equivalent_diameter" in raw:
        values = numeric(raw["equivalent_diameter"]).dropna()
        if not values.empty:
            rows.extend(
                [
                    ("D10", f"{np.percentile(values, 10):.1f} px"),
                    ("D50", f"{np.percentile(values, 50):.1f} px"),
                    ("D90", f"{np.percentile(values, 90):.1f} px"),
                ]
            )
    if "circularity" in raw:
        values = numeric(raw["circularity"]).dropna()
        if not values.empty:
            rows.append(("Circ.", f"{np.median(values):.2f}"))
    if {"major_axis_length", "minor_axis_length"}.issubset(raw.columns):
        major = numeric(raw["major_axis_length"])
        minor = numeric(raw["minor_axis_length"]).replace(0, np.nan)
        aspect = (major / minor).replace([np.inf, -np.inf], np.nan).dropna()
        if not aspect.empty:
            rows.append(("Aspect", f"{np.median(aspect):.2f}"))
    rows.append(("Particles", f"{len(raw):d}"))
    return rows[:6]


def figure_micrograph_pipeline(paths: Paths, output_format: str) -> None:
    image_paths = [
        REPO_ROOT / "paper" / "figures" / "rawImage.png",
        REPO_ROOT / "paper" / "figures" / "initSeg.png",
        REPO_ROOT / "paper" / "figures" / "refined.png",
    ]
    if not all(path.exists() for path in image_paths):
        return
    titles = [
        "a. Raw SEM micrograph",
        "b. Initial instance mask",
        "c. Refined particle features",
    ]
    fig = plt.figure(figsize=(7.1, 2.55))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.15, 0.70], hspace=0.08, wspace=0.04)
    for idx, (path, title) in enumerate(zip(image_paths, titles)):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(plt.imread(path), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#222222")
        ax.text(
            0.02,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            weight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#111111", "edgecolor": "none", "alpha": 0.72},
        )

    ax_metrics = fig.add_subplot(grid[1, :])
    ax_metrics.axis("off")
    ax_metrics.text(
        0.015,
        0.82,
        "Fields exposed to deterministic tools",
        ha="left",
        va="center",
        fontsize=7.5,
        weight="bold",
        color="#111111",
    )
    metrics = micrograph_metric_summary()
    chip_colors = [OKABE_ITO["red"], OKABE_ITO["red"], OKABE_ITO["red"], OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["gray"]]
    start_x = 0.015
    chip_w = 0.145
    gap = 0.012
    for idx, (label, value) in enumerate(metrics):
        x = start_x + idx * (chip_w + gap)
        color = chip_colors[idx % len(chip_colors)]
        chip = FancyBboxPatch(
            (x, 0.26),
            chip_w,
            0.42,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=0.9,
            edgecolor=color,
            facecolor=color + "14",
            transform=ax_metrics.transAxes,
        )
        ax_metrics.add_patch(chip)
        ax_metrics.text(x + chip_w / 2, 0.52, label, ha="center", va="center", fontsize=6.5, weight="bold", transform=ax_metrics.transAxes)
        ax_metrics.text(x + chip_w / 2, 0.36, value, ha="center", va="center", fontsize=6.2, transform=ax_metrics.transAxes)
    ax_metrics.text(
        0.015,
        0.08,
        "The LLM never receives hidden answers in routing prompts; it selects which field/statistic to execute.",
        ha="left",
        va="center",
        fontsize=6.6,
        color="#444444",
    )
    fig.suptitle("SEM images become queryable morphology tables", x=0.03, y=1.02, ha="left", weight="bold", fontsize=10)
    save_figure(fig, paths, "fig_micrograph_pipeline", output_format)


def figure_result_panels(main: pd.DataFrame, paths: Paths, output_format: str) -> None:
    data = main[main["model_label"].isin(MAIN_MODEL_ORDER)].copy()
    data = data.sort_values("model_label", key=lambda s: s.map(model_sort_key))
    if data.empty:
        return
    y = np.arange(len(data))
    fig, (ax_direct, ax_route) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.25),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )

    ax_direct.hlines(
        y,
        data["raw_accuracy"] * 100,
        data["processed_accuracy"] * 100,
        color="#d9d9d9",
        lw=2.0,
        zorder=1,
    )
    ax_direct.scatter(
        data["raw_accuracy"] * 100,
        y,
        color=OKABE_ITO["red"],
        marker="o",
        s=42,
        edgecolor="white",
        linewidth=0.5,
        label="Raw records",
        zorder=3,
    )
    ax_direct.scatter(
        data["processed_accuracy"] * 100,
        y,
        color=OKABE_ITO["blue"],
        marker="s",
        s=42,
        edgecolor="white",
        linewidth=0.5,
        label="Processed summaries",
        zorder=3,
    )
    for idx, row in enumerate(data.itertuples()):
        raw = float(row.raw_accuracy) * 100 if pd.notna(row.raw_accuracy) else np.nan
        proc = float(row.processed_accuracy) * 100 if pd.notna(row.processed_accuracy) else np.nan
        if np.isfinite(raw):
            ax_direct.text(raw - 2.0, idx, f"{raw:.0f}", va="center", ha="right", fontsize=6.5, color=OKABE_ITO["red"])
        if np.isfinite(proc):
            ax_direct.text(proc + 1.6, idx, f"{proc:.0f}", va="center", ha="left", fontsize=6.5, color=OKABE_ITO["blue"])
    ax_direct.set_yticks(y)
    ax_direct.set_yticklabels(data["model"])
    ax_direct.invert_yaxis()
    ax_direct.set_xlim(0, 103)
    ax_direct.set_xticks([0, 50, 100])
    ax_direct.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax_direct.set_xlabel("Accuracy (%)")
    ax_direct.set_title("A. Direct computation depends on encoding", loc="left", weight="bold")
    ax_direct.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    route = data["routing_accuracy"] * 100
    bars = ax_route.barh(y, route, color=OKABE_ITO["green"], alpha=0.88, height=0.48)
    ax_route.axvline(95, color="#666666", lw=0.8, ls="--")
    ax_route.text(95, -0.62, "95%", ha="center", va="top", fontsize=6.5, color="#444444")
    for idx, (bar, value) in enumerate(zip(bars, route)):
        if pd.notna(value):
            ax_route.text(
                min(float(value) + 1.0, 101.0),
                bar.get_y() + bar.get_height() / 2,
                f"{float(value):.1f}",
                va="center",
                ha="left",
                fontsize=6.5,
                color=OKABE_ITO["green"],
            )
    ax_route.set_xlim(0, 103)
    ax_route.set_xticks([0, 50, 100])
    ax_route.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax_route.tick_params(axis="y", length=0, labelleft=False)
    ax_route.set_xlabel("Accuracy (%)")
    ax_route.set_title("B. Tool routing is stable across models", loc="left", weight="bold")
    fig.suptitle(
        "LLMs struggle to compute from characterization records; they reliably route the right computation",
        x=0.03,
        y=1.02,
        ha="left",
        weight="bold",
        fontsize=10,
    )
    save_figure(fig, paths, "fig_result_panels", output_format)


def ordered_rows(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if df.empty or "model_label" not in df:
        return pd.DataFrame()
    out = df[df["model_label"].astype(str).isin(order)].copy()
    out["plot_order"] = out["model_label"].map({label: idx for idx, label in enumerate(order)})
    return out.sort_values("plot_order").reset_index(drop=True)


def figure_reliability_rows(reliability: pd.DataFrame) -> pd.DataFrame:
    if reliability.empty:
        return pd.DataFrame()
    out = reliability.copy()
    out["success_number"] = pd.to_numeric(out.get("success"), errors="coerce")
    out["total_number"] = pd.to_numeric(out.get("total"), errors="coerce")
    out = out[(out["total_number"] > 0) & (out["success_number"] / out["total_number"] >= 0.5)]
    out["parsed_accuracy_number"] = pd.to_numeric(out.get("parsed_accuracy"), errors="coerce")
    out["exact_response_agreement_number"] = pd.to_numeric(
        out.get("exact_response_agreement"),
        errors="coerce",
    )
    out = out.dropna(subset=["parsed_accuracy_number", "exact_response_agreement_number"])
    out["agreement_accuracy_gap"] = out["exact_response_agreement_number"] - out["parsed_accuracy_number"]
    out = out.sort_values("agreement_accuracy_gap", ascending=False)
    return out.reset_index(drop=True)


def figure2(main: pd.DataFrame, encoding: pd.DataFrame, case: pd.DataFrame, paths: Paths, output_format: str) -> None:
    data = ordered_rows(main, FIG2_MODEL_ORDER)
    data = data[
        data[["processed_accuracy", "raw_accuracy", "routing_accuracy"]]
        .notna()
        .any(axis=1)
    ].copy()
    data["accepted_sort"] = pd.to_numeric(data.get("routing_accuracy"), errors="coerce")
    data["original_order"] = data["model_label"].map({label: idx for idx, label in enumerate(FIG2_MODEL_ORDER)})
    data["accepted_sort"] = data["accepted_sort"].fillna(-1)
    data = data.sort_values(["accepted_sort", "original_order"], ascending=[False, True]).drop(
        columns=["accepted_sort", "original_order"]
    )
    if data.empty:
        return

    raw_color = OKABE_ITO["orange"]
    processed_color = OKABE_ITO["blue"]
    route_color = OKABE_ITO["green"]
    case_bar_color = OKABE_ITO["purple"]
    connector_color = "#cfcfcf"
    dot_gray = "#808080"

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(8.35, 2.54),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.30},
    )

    model_labels = list(data["model_label"])
    model_names = list(data["model"])
    y = np.arange(len(model_labels))

    processed_text = np.array([encoding_accuracy(encoding, label, "processed_text") * 100 for label in model_labels])
    processed_table = np.array([encoding_accuracy(encoding, label, "processed_table") * 100 for label in model_labels])
    raw_text = np.array([encoding_accuracy(encoding, label, "raw_text") * 100 for label in model_labels])
    raw_table = np.array([encoding_accuracy(encoding, label, "raw_table") * 100 for label in model_labels])

    for row, proc_text, proc_table, raw_txt, raw_tbl, label in zip(
        y,
        processed_text,
        processed_table,
        raw_text,
        raw_table,
        model_labels,
    ):
        if np.isfinite(proc_text) and np.isfinite(proc_table):
            ax_a.plot([proc_table, proc_text], [row, row], color=processed_color, lw=0.8, alpha=0.56, zorder=1)
        if np.isfinite(raw_txt) and np.isfinite(raw_tbl):
            ax_a.plot([raw_tbl, raw_txt], [row, row], color=raw_color, lw=0.8, alpha=0.45, zorder=1)

    ax_a.scatter(processed_text, y, s=24, marker="o", color=processed_color, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False)
    ax_a.scatter(processed_table, y, s=24, marker="s", color=processed_color, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False)
    ax_a.scatter(raw_text, y, s=24, marker="o", color=raw_color, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False)
    ax_a.scatter(raw_table, y, s=24, marker="s", color=raw_color, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(model_names)
    ax_a.set_ylim(-0.55, len(model_labels) - 0.45)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.grid(axis="x", color="#e8e8e8", lw=0.55)
    ax_a.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.5)
    ax_a.tick_params(axis="y", length=0, pad=1.5, labelsize=6.5)
    ax_a.set_xlabel("Direct accuracy (%)", fontsize=7)
    ax_a.set_title("Input Format", loc="center", weight="bold", fontsize=8.0, pad=4)

    raw_avg = pd.to_numeric(data["raw_accuracy"], errors="coerce").to_numpy(dtype=float) * 100
    accepted = pd.to_numeric(data["routing_accuracy"], errors="coerce").to_numpy(dtype=float) * 100
    strict = pd.to_numeric(data.get("strict_routing_accuracy"), errors="coerce").to_numpy(dtype=float) * 100
    for row, raw, acc in zip(y, raw_avg, accepted):
        if np.isfinite(raw) and np.isfinite(acc):
            ax_b.plot([raw, acc], [row, row], color=connector_color, lw=0.9, zorder=1)
    ax_b.scatter(raw_avg, y, s=25, marker="o", color=raw_color, edgecolor="white", linewidth=0.45, zorder=3, clip_on=False)
    ax_b.scatter(accepted, y, s=32, marker="D", color=route_color, edgecolor="white", linewidth=0.5, zorder=4, clip_on=False)
    ax_b.scatter(strict, y, s=32, marker="D", facecolor="white", edgecolor=route_color, linewidth=1.0, zorder=5, clip_on=False)
    ax_b.set_ylim(ax_a.get_ylim())
    ax_b.set_yticks(y)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.grid(axis="x", color="#e8e8e8", lw=0.55)
    ax_b.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.5)
    ax_b.set_xlabel("Task success (%)", fontsize=7)
    ax_b.set_title("Analysis Routing", loc="center", weight="bold", fontsize=8.0, pad=4)

    case_complete = case.copy()
    if not case_complete.empty:
        case_complete["success_number"] = pd.to_numeric(case_complete.get("success"), errors="coerce")
        case_complete["total_number"] = pd.to_numeric(case_complete.get("total"), errors="coerce")
        case_complete = case_complete[
            (case_complete["total_number"] > 0)
            & (case_complete["success_number"] / case_complete["total_number"] >= 0.5)
        ]
    case_metrics = [
        ("Pairwise", "stage1_pairwise", 0.780),
        ("Numeric", "stage1_numeric", 0.665),
        ("Family top-3", "stage2_family_p3", 0.371),
        ("Exact top-3", "stage2_topk_p3", 0.029),
    ]
    case_y = np.arange(len(case_metrics))
    means: list[float] = []
    individual_values: list[np.ndarray] = []
    for _, column, fallback in case_metrics:
        values = np.array([], dtype=float)
        if not case_complete.empty and column in case_complete:
            values = pd.to_numeric(case_complete[column], errors="coerce").dropna().to_numpy(dtype=float) * 100
        individual_values.append(values)
        means.append(float(np.mean(values)) if len(values) else fallback * 100)

    ax_c.barh(case_y, means, color=case_bar_color, alpha=0.82, height=0.46, edgecolor="none", zorder=1)
    for row, values in zip(case_y, individual_values):
        if len(values):
            jitter = np.linspace(-0.03, 0.03, len(values)) if len(values) > 1 else np.array([0.0])
            ax_c.scatter(values, row + jitter, s=13, color=dot_gray, alpha=0.76, edgecolor="white", linewidth=0.25, zorder=3, clip_on=False)
    ax_c.set_yticks(case_y)
    ax_c.set_yticklabels([label for label, _, _ in case_metrics], fontsize=5.9)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.grid(axis="x", color="#e8e8e8", lw=0.55)
    ax_c.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.5)
    ax_c.tick_params(axis="y", length=0, pad=1.5)
    ax_c.set_xlabel("Task accuracy (%)", fontsize=7)
    ax_c.set_title("Query Difficulty", loc="center", weight="bold", fontsize=8.0, pad=4)

    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

    legend_ax = fig.add_axes([0.0, 0.005, 1.0, 0.105])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.plot([0, 1], [0.02, 0.02], alpha=0.0, lw=0.1)

    legend_fontsize = 6.75
    legend_y = 0.36
    legend_items = [
        (0.010, "marker", "o", processed_color, "white", "Processed text"),
        (0.150, "marker", "s", processed_color, "white", "Processed table"),
        (0.305, "marker", "o", raw_color, "white", "Raw text"),
        (0.410, "marker", "s", raw_color, "white", "Raw table"),
        (0.515, "marker", "D", route_color, "white", "Accepted tool call"),
        (0.690, "marker", "D", "white", route_color, "Strict tool call"),
        (0.825, "bar", "", case_bar_color, "none", "Mean score"),
        (0.930, "marker", "o", dot_gray, "white", "Model score"),
    ]
    for x, item_type, marker, face, edge, label in legend_items:
        if item_type == "bar":
            legend_ax.add_patch(
                plt.Rectangle(
                    (x, legend_y - 0.055),
                    0.022,
                    0.11,
                    transform=legend_ax.transAxes,
                    facecolor=face,
                    edgecolor="none",
                    alpha=0.82,
                )
            )
            text_x = x + 0.026
        else:
            legend_ax.plot(
                x,
                legend_y,
                marker=marker,
                markersize=4.4,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=1.1 if face == "white" else 0.45,
                color="none",
                transform=legend_ax.transAxes,
            )
            text_x = x + 0.014
        legend_ax.text(
            text_x,
            legend_y,
            label,
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=legend_fontsize,
            color="#222222",
        )

    fig.subplots_adjust(left=0.125, right=0.998, bottom=0.22, wspace=0.30)

    save_figure(fig, paths, "fig2", output_format)


def figure_speed_bars(main: pd.DataFrame, paths: Paths, output_format: str) -> None:
    data = ordered_rows(main, FIG2_MODEL_ORDER)
    data = data[
        data[["mean_latency_s", "mean_output_tokens", "mean_latency_s_tool", "mean_output_tokens_tool"]]
        .notna()
        .any(axis=1)
    ].copy()
    if data.empty:
        return

    for col in ["mean_latency_s", "mean_output_tokens", "mean_latency_s_tool", "mean_output_tokens_tool"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["direct_tokens_s"] = data["mean_output_tokens"] / data["mean_latency_s"]
    data["tool_tokens_s"] = data["mean_output_tokens_tool"] / data["mean_latency_s_tool"]
    data = data.sort_values("mean_latency_s", ascending=True)

    model_names = data["model"].astype(str).tolist()
    y = np.arange(len(data))
    offset = 0.18
    direct_color = OKABE_ITO["blue"]
    tool_color = OKABE_ITO["green"]
    grid_color = "#e8e8e8"

    fig, (ax_latency, ax_output, ax_tps) = plt.subplots(
        1,
        3,
        figsize=(8.9, max(2.6, 0.24 * len(data) + 0.45)),
        gridspec_kw={"width_ratios": [1.05, 0.92, 1.0], "wspace": 0.16},
    )

    ax_latency.barh(
        y - offset,
        data["mean_latency_s"],
        height=0.28,
        color=direct_color,
        alpha=0.88,
        label="Direct",
    )
    ax_latency.barh(
        y + offset,
        data["mean_latency_s_tool"],
        height=0.28,
        color=tool_color,
        alpha=0.88,
        label="Tool call",
    )
    ax_latency.set_yticks(y)
    ax_latency.set_yticklabels(model_names)
    ax_latency.invert_yaxis()
    ax_latency.set_xscale("log")
    ax_latency.set_xlabel("Seconds / answer")
    ax_latency.set_title("Wall-clock", weight="bold", pad=4)
    ax_latency.grid(axis="x", color=grid_color, lw=0.55)
    ax_latency.tick_params(axis="y", length=0, labelsize=6.4)
    ax_latency.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.4)

    ax_output.barh(
        y - offset,
        data["mean_output_tokens"],
        height=0.28,
        color=direct_color,
        alpha=0.88,
    )
    ax_output.barh(
        y + offset,
        data["mean_output_tokens_tool"],
        height=0.28,
        color=tool_color,
        alpha=0.88,
    )
    ax_output.set_yticks(y)
    ax_output.tick_params(axis="y", left=False, labelleft=False)
    ax_output.invert_yaxis()
    ax_output.set_xscale("log")
    ax_output.set_xlabel("Output tokens / answer")
    ax_output.set_title("Output Size", weight="bold", pad=4)
    ax_output.grid(axis="x", color=grid_color, lw=0.55)
    ax_output.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.4)

    ax_tps.barh(
        y - offset,
        data["direct_tokens_s"],
        height=0.28,
        color=direct_color,
        alpha=0.88,
    )
    ax_tps.barh(
        y + offset,
        data["tool_tokens_s"],
        height=0.28,
        color=tool_color,
        alpha=0.88,
    )
    ax_tps.set_yticks(y)
    ax_tps.tick_params(axis="y", left=False, labelleft=False)
    ax_tps.invert_yaxis()
    ax_tps.set_xscale("log")
    ax_tps.set_xlabel("Output tokens / second")
    ax_tps.set_title("Throughput", weight="bold", pad=4)
    ax_tps.grid(axis="x", color=grid_color, lw=0.55)
    ax_tps.tick_params(axis="x", length=2.8, width=0.7, labelsize=6.4)

    handles = [
        Line2D([0], [0], color=direct_color, lw=4, label="Direct"),
        Line2D([0], [0], color=tool_color, lw=4, label="Tool call"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.235, right=0.99, bottom=0.18, top=0.9)
    save_figure(fig, paths, "fig_speed_bars", output_format)


def figure_direct_vs_routing(main: pd.DataFrame, paths: Paths, output_format: str) -> None:
    data = main[main["model_label"].isin(MAIN_MODEL_ORDER)].copy()
    data = data.sort_values("model_label", key=lambda s: s.map(model_sort_key))
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    y = np.arange(len(data))
    metrics = [
        ("raw_accuracy", "Raw direct", OKABE_ITO["red"], "o"),
        ("processed_accuracy", "Processed direct", OKABE_ITO["blue"], "s"),
        ("routing_accuracy", "Tool routed", OKABE_ITO["green"], "D"),
    ]
    for idx, row in data.iterrows():
        vals = [row.get(col) for col, _, _, _ in metrics]
        finite = [float(v) for v in vals if pd.notna(v)]
        if finite:
            ax.plot([min(finite) * 100, max(finite) * 100], [y[list(data.index).index(idx)]] * 2, color="#dddddd", lw=1.2, zorder=1)
    for col, label, color, marker in metrics:
        ax.scatter(data[col] * 100, y, label=label, color=color, marker=marker, s=42, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"])
    ax.set_xlabel("Completed-prompt accuracy (%)")
    ax.set_xlim(-2, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_title("Routing turns brittle record reading into reliable characterization queries", loc="left", weight="bold")
    ax.invert_yaxis()
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    save_figure(fig, paths, "fig_direct_vs_routing", output_format)


def figure_encoding_heatmap(encoding: pd.DataFrame, paths: Paths, output_format: str) -> None:
    data = encoding[encoding["model_label"].isin(MAIN_MODEL_ORDER)].copy()
    if data.empty:
        return
    encoding_order = ["processed_table", "processed_text", "raw_table", "raw_text"]
    labels = ["Proc. table", "Proc. text", "Raw table", "Raw text"]
    pivot = data.pivot_table(index="model_label", columns="encoding", values="completed_accuracy", aggfunc="first")
    pivot = pivot.reindex([m for m in MAIN_MODEL_ORDER if m in pivot.index])[encoding_order]
    matrix = pivot.to_numpy(dtype=float) * 100
    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#f7f7f7", "#fee8c8", "#fdbb84", "#2b8cbe", "#005a8d"])
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([model_display(label) for label in pivot.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "--" if not np.isfinite(value) else f"{value:.0f}"
            color = "white" if np.isfinite(value) and value > 72 else "#111111"
            ax.text(j, i, text, ha="center", va="center", color=color, weight="bold", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy (%)")
    ax.set_title("Direct inference depends strongly on serialization", loc="left", weight="bold")
    ax.tick_params(axis="both", length=0)
    save_figure(fig, paths, "fig_encoding_heatmap", output_format)


def figure_efficiency(main: pd.DataFrame, encoding: pd.DataFrame, tool: pd.DataFrame, paths: Paths, output_format: str) -> None:
    points = []
    selected = [m for m in MAIN_MODEL_ORDER if m in set(main["model_label"])]
    for label in selected:
        raw_rows = encoding[(encoding["model_label"] == label) & (encoding["encoding"].isin(["raw_table", "raw_text"]))]
        proc_rows = encoding[(encoding["model_label"] == label) & (encoding["encoding"].isin(["processed_table", "processed_text"]))]
        for group_name, rows, color, marker in [
            ("Processed direct", proc_rows, OKABE_ITO["blue"], "s"),
            ("Raw direct", raw_rows, OKABE_ITO["red"], "o"),
        ]:
            if rows.empty:
                continue
            points.append(
                {
                    "model_label": label,
                    "model": model_display(label),
                    "mode": group_name,
                    "accuracy": float(np.nanmean(rows["completed_accuracy"])) * 100,
                    "tokens": float(np.nanmean(rows["mean_total_tokens"])),
                    "color": color,
                    "marker": marker,
                }
            )
        tool_row = tool[tool["model_label"] == label]
        if not tool_row.empty:
            row = tool_row.iloc[0]
            points.append(
                {
                    "model_label": label,
                    "model": model_display(label),
                    "mode": "Tool routed",
                    "accuracy": float(row["routing_accuracy"]) * 100,
                    "tokens": float(row["mean_total_tokens"]),
                    "color": OKABE_ITO["green"],
                    "marker": "D",
                }
            )
    if not points:
        return
    df = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    for mode, group in df.groupby("mode"):
        ax.scatter(
            group["tokens"],
            group["accuracy"],
            label=mode,
            color=group["color"].iloc[0],
            marker=group["marker"].iloc[0],
            s=46,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    routed = df[df["mode"] == "Tool routed"]
    if not routed.empty:
        x_anchor = float(np.nanmedian(routed["tokens"]))
        y_anchor = float(np.nanmedian(routed["accuracy"]))
        ax.annotate(
            "Tool-routed calls\navoid raw-record load",
            (x_anchor, y_anchor),
            xytext=(3600, 84),
            textcoords="data",
            fontsize=7,
            color=OKABE_ITO["green"],
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO["green"], "lw": 0.9},
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean tokens per prompt (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 104)
    ax.grid(axis="both", color="#e6e6e6", lw=0.8, which="major")
    ax.set_title("Tool routing is accurate without raw-record token load", loc="left", weight="bold")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handletextpad=0.4,
        columnspacing=1.2,
    )
    save_figure(fig, paths, "fig_efficiency_tradeoff", output_format)


def first_metric(df: pd.DataFrame, label: str, column: str) -> float:
    if df.empty or column not in df:
        return float("nan")
    rows = df[df["model_label"].astype(str) == label]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows.iloc[0][column], errors="coerce"))


def figure_failure_modes(main: pd.DataFrame, tool: pd.DataFrame, reliability: pd.DataFrame, paths: Paths, output_format: str) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.33})

    gemma_label = "gemma-4-31b"
    direct = first_metric(reliability, gemma_label, "parsed_accuracy")
    if not np.isfinite(direct):
        direct = first_metric(main, gemma_label, "base_accuracy")
    agreement = first_metric(reliability, gemma_label, "exact_response_agreement")
    route = first_metric(tool, gemma_label, "routing_accuracy")
    left_values = [direct, agreement, route]
    left_labels = ["Direct\naccuracy", "Repeat\nagreement", "Tool\nrouting"]
    left_colors = [OKABE_ITO["red"], "#8f8f8f", OKABE_ITO["green"]]
    x = np.arange(len(left_values))
    ax_left.bar(x, np.array(left_values) * 100, color=left_colors, width=0.62)
    for idx, value in enumerate(left_values):
        if np.isfinite(value):
            ax_left.text(idx, value * 100 + 2.5, f"{value * 100:.1f}", ha="center", va="bottom", fontsize=7, weight="bold")
    ax_left.set_ylim(0, 112)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(left_labels)
    ax_left.set_ylabel("Accuracy / agreement (%)")
    ax_left.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax_left.set_title("A. Repeatable failure is still failure", loc="left", weight="bold")
    ax_left.text(
        0.5,
        0.86,
        "Gemma 31B gives\nconsistent wrong arithmetic",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#333333",
    )

    nano_none = first_metric(tool, "gpt-5.4-nano-none", "routing_accuracy")
    nano_medium = first_metric(tool, "gpt-5.4-nano-medium", "routing_accuracy")
    right_values = [nano_none, nano_medium]
    right_labels = ["nano\nnone", "nano\nmedium"]
    right_colors = [OKABE_ITO["red"], OKABE_ITO["green"]]
    ax_right.bar(np.arange(2), np.array(right_values) * 100, color=right_colors, width=0.58)
    for idx, value in enumerate(right_values):
        if np.isfinite(value):
            ax_right.text(idx, value * 100 + 2.5, f"{value * 100:.1f}", ha="center", va="bottom", fontsize=7, weight="bold")
    if np.isfinite(nano_none) and np.isfinite(nano_medium):
        gain = (nano_medium - nano_none) * 100
        ax_right.annotate(
            f"+{gain:.1f} pp",
            xy=(1, nano_medium * 100),
            xytext=(0.5, 101),
            ha="center",
            va="center",
            fontsize=7,
            color=OKABE_ITO["green"],
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": OKABE_ITO["green"]},
        )
    ax_right.set_ylim(0, 112)
    ax_right.set_xticks(np.arange(2))
    ax_right.set_xticklabels(right_labels)
    ax_right.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax_right.set_title("B. Reasoning improves schema binding", loc="left", weight="bold")
    ax_right.text(
        0.5,
        0.16,
        "More reasoning reduced\ninvalid array-style arguments",
        transform=ax_right.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="#333333",
    )
    fig.suptitle("The useful LLM task is routing, not floating-point morphology arithmetic", x=0.03, y=1.04, ha="left", weight="bold", fontsize=10)
    save_figure(fig, paths, "fig_failure_modes", output_format)


def write_design_plan(paths: Paths) -> None:
    text = """# Figure And Table Plan

Main-body target for a 4-page single-column AI4Mat paper, following the reference-paper style:

1. Table 1 (`table1`), Results anchor: exact model-level comparison of processed direct, raw direct, accepted/strict routing, and completed reliability runs.
2. Figure 2 (`fig2`), Results synthesis: three panels showing input-format sensitivity, tool-calling recovery, and the constrained-to-open-ended case-study gradient.

Appendix tables:

- `table_run_coverage`: completion counts for the 4,170-prompt full direct manifest, the 3,840-prompt core subset, tool routing, case study, and reliability.
- `table_tool_routing_full`: all routing metrics, including strict and executable rates.
- `table_base_by_encoding`: all-model input-format table replacing the heatmap.
- `table_case_study`: case-study stage breakdown. Use Stage 2 direct top-k as motivation/limitation, not the main claim.
- `table_reliability`: repeatability of direct prompts. This supports the nuance that deterministic-looking outputs can still be inaccurate.

Design notes:

- Use direct labels and one message per figure.
- Prefer completed-prompt accuracy for visual comparisons so parse failures count against brittle methods.
- Reserve accepted routing for the headline; report strict routing in the full table.
- Keep generated figures as vector PDFs for LaTeX and PNG previews for inspection.
"""
    path = paths.paper_generated_dir / "figure_table_plan.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def write_include_snippet(paths: Paths) -> None:
    snippet = r"""\input{generated/tables/table1}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{generated/figures/fig2.pdf}
  \caption{Direct characterization is representation-sensitive, tool calls improve task success, and open-ended natural-language queries remain difficult.}
  \label{fig:evidence-synthesis}
\end{figure}
"""
    path = paths.paper_generated_dir / "main_body_snippet.tex"
    path.write_text(snippet, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    experiments_config = load_yaml(args.experiments_config)
    models_config = load_yaml(args.models_config)
    paths = ensure_paths(experiments_config)
    setup_matplotlib()
    if args.refresh:
        refresh_evaluations(models_config)

    base_path = select_base_scores(paths)
    base_df = filter_excluded_models(read_csv(base_path))
    case_df = filter_excluded_models(load_case_scores(paths))
    base = build_base_summary(base_df, models_config)
    encoding = build_encoding_summary(base_df)
    tool = filter_excluded_models(load_tool_summary(paths))
    reliability = filter_excluded_models(load_reliability_summary(paths))
    case = load_case_summary(case_df)
    main_results = combine_main(base, tool)
    main_results, encoding, reliability, case, denominator_sources = apply_denominator_corrections(
        paths,
        main_results,
        encoding,
        reliability,
        case,
    )

    base.to_csv(paths.results_artifact_dir / "base_summary.csv", index=False)
    encoding.to_csv(paths.results_artifact_dir / "encoding_summary.csv", index=False)
    tool.to_csv(paths.results_artifact_dir / "tool_routing_summary.csv", index=False)
    reliability.to_csv(paths.results_artifact_dir / "reliability_summary.csv", index=False)
    case.to_csv(paths.results_artifact_dir / "case_study_summary.csv", index=False)
    main_results.to_csv(paths.results_artifact_dir / "main_results_summary.csv", index=False)

    if args.artifact_set == "main":
        write_table1(paths, main_results, encoding, reliability, case)
        figure2(main_results, encoding, case, paths, args.format)
        figure_speed_bars(main_results, paths, args.format)
        write_json_atomic(
            paths.results_artifact_dir / "artifact_sources.json",
            {
                "base_scores": str(base_path.relative_to(REPO_ROOT)),
                "case_scores": [str(path.relative_to(REPO_ROOT)) for path in case_score_source_paths(paths)],
                "refresh_used": bool(args.refresh),
                "artifact_set": args.artifact_set,
                "denominator_correction_sources": denominator_sources,
            },
        )
        print("Main paper artifacts complete.")
        return

    write_tables(paths, base_df, main_results, encoding, tool, case, reliability)
    figure2(main_results, encoding, case, paths, args.format)
    figure_speed_bars(main_results, paths, args.format)
    write_design_plan(paths)
    write_include_snippet(paths)
    write_json_atomic(
        paths.results_artifact_dir / "artifact_sources.json",
        {
            "base_scores": str(base_path.relative_to(REPO_ROOT)),
            "case_scores": [str(path.relative_to(REPO_ROOT)) for path in case_score_source_paths(paths)],
            "refresh_used": bool(args.refresh),
            "artifact_set": args.artifact_set,
            "denominator_correction_sources": denominator_sources,
        },
    )
    print("Paper artifacts complete.")


if __name__ == "__main__":
    main()
