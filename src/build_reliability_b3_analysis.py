#!/usr/bin/env python3
"""Build appendix B.3 repeated-prompt reliability analyses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai4mat_common import REPO_ROOT, slugify, write_json_atomic


EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"
RAW_DIR = REPO_ROOT / "results" / "raw"
OUTPUT_DIR = REPO_ROOT / "paper" / "generated" / "potential"
EXPECTED_RESPONSES = 768
EXPECTED_GROUPS = 256
LENGTH_REASONS = {"length", "max_tokens", "max_output_tokens"}

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
MODEL_CLASS = {
    "gpt-5.6-luna-none": "GPT series / API",
    "gpt-5.6-luna-medium": "GPT series / API",
    "gpt-5.4-mini-none": "GPT series / API",
    "gpt-5.4-mini-medium": "GPT series / API",
    "gpt-5.4-nano-none": "GPT series / API",
    "gpt-5.4-nano-medium": "GPT series / API",
}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{100.0 * number:.1f}"


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


def display_name(label: str) -> str:
    return MODEL_NAMES.get(label, label)


def model_class(label: str) -> str:
    return MODEL_CLASS.get(label, "Open-weight / local")


def model_sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {label: idx for idx, label in enumerate(MODEL_ORDER)}
    out = df.copy()
    out["_order"] = out["model_label"].map(order).fillna(len(order))
    return out.sort_values(["_order", "model_label"]).drop(columns="_order").reset_index(drop=True)


def read_finish_reason(model_label: str, run_id: str, task_id: str) -> str:
    path = RAW_DIR / slugify(model_label) / slugify(run_id) / f"{task_id}.json"
    if not path.exists():
        return "missing_output_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable_output_file"
    return str(payload.get("finish_reason", "")).strip().lower()


def load_scores() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label in MODEL_ORDER:
        path = EVALUATION_DIR / f"paper_reliability_{label}_scores.csv"
        if not path.exists() or label in EXCLUDED_MODELS:
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["model_label"] = label
        frames.append(frame)
    if not frames:
        raise SystemExit("No paper_reliability_* score files found.")
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[~scores["model_label"].astype(str).isin(EXCLUDED_MODELS)].copy()
    scores["model"] = scores["model_label"].map(display_name)
    scores["model_class"] = scores["model_label"].map(model_class)
    scores["response_norm"] = scores["response"].map(normalized_text)
    scores["prediction_norm"] = scores["prediction"].map(normalized_text)
    scores["is_success"] = scores["status"].astype(str).eq("success")
    scores["is_empty_response"] = scores["response"].fillna("").astype(str).str.strip().eq("")
    scores["is_completed"] = scores["is_success"] & ~scores["is_empty_response"]
    scores["is_parsed"] = scores["is_completed"] & truthy(scores["parse_success"])
    scores["is_correct_e2e"] = scores["is_success"] & truthy(scores["parse_success"]) & truthy(scores["correct"])
    scores["finish_reason"] = [
        read_finish_reason(str(row.model_label), str(row.source_run_id), str(row.task_id))
        for row in scores.itertuples(index=False)
    ]
    scores["is_length_stop"] = scores["finish_reason"].isin(LENGTH_REASONS)
    scores["task_category"] = scores.apply(task_category, axis=1)
    return scores


def task_category(row: pd.Series) -> str:
    suite = str(row.get("suite", ""))
    encoding = str(row.get("encoding", ""))
    answer_type = str(row.get("answer_type", ""))
    if suite == "core":
        return {
            "raw_table": "Raw table",
            "raw_text": "Raw text",
            "processed_table": "Processed table",
            "processed_text": "Processed text",
        }.get(encoding, "Core other")
    if suite == "interface_numeric" or answer_type == "numeric":
        return "Materials numeric"
    if suite == "interface_pairwise" or answer_type == "label":
        return "Pairwise"
    return suite or "Other"


def group_records(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, group_id), group in scores.groupby(["model_label", "repeat_group_id"], dropna=False):
        if not group_id or str(group_id) == "nan":
            continue
        group = group.sort_values("repeat_index")
        parsed_predictions = group.loc[group["is_parsed"], "prediction_norm"].tolist()
        exact_agree = group["response_norm"].nunique(dropna=False) <= 1
        parsed_agree = bool(parsed_predictions) and len(set(parsed_predictions)) == 1
        correct_count = int(group["is_correct_e2e"].sum())
        same_correct = bool(exact_agree and len(group) == 3 and correct_count == 3)
        same_wrong = bool(exact_agree and len(group) == 3 and correct_count == 0)
        same_mixed = bool(exact_agree and len(group) == 3 and 0 < correct_count < 3)
        disagreed = not exact_agree
        first = group.iloc[0]
        rows.append(
            {
                "model_label": str(model_label),
                "model": display_name(str(model_label)),
                "model_class": model_class(str(model_label)),
                "repeat_group_id": group_id,
                "suite": first.get("suite", ""),
                "answer_type": first.get("answer_type", ""),
                "encoding": first.get("encoding", ""),
                "feature": first.get("feature", ""),
                "statistic": first.get("statistic", ""),
                "task_category": first.get("task_category", ""),
                "repeats_observed": int(len(group)),
                "completed_count": int(group["is_completed"].sum()),
                "parsed_count": int(group["is_parsed"].sum()),
                "correct_count": correct_count,
                "exact_response_agreement": exact_agree,
                "parsed_answer_agreement": parsed_agree,
                "same_correct": same_correct,
                "same_wrong": same_wrong,
                "same_mixed": same_mixed,
                "disagreed": disagreed,
            }
        )
    return model_sort(pd.DataFrame(rows))


def model_summary(scores: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        model_rows = scores[scores["model_label"].eq(label)]
        group_rows = groups[groups["model_label"].eq(label)]
        if model_rows.empty:
            continue
        parsed = model_rows["is_parsed"].sum()
        rows.append(
            {
                "model_label": label,
                "model": display_name(label),
                "model_class": model_class(label),
                "completed_n": int(model_rows["is_completed"].sum()),
                "completed_of": EXPECTED_RESPONSES,
                "parsed_n": int(parsed),
                "parsed_of": EXPECTED_RESPONSES,
                "overall_repeat_accuracy": float(model_rows["is_correct_e2e"].mean()),
                "parsed_only_accuracy": float(model_rows.loc[model_rows["is_parsed"], "is_correct_e2e"].mean())
                if parsed
                else float("nan"),
                "exact_response_agreement": float(group_rows["exact_response_agreement"].mean()),
                "parsed_answer_agreement": float(group_rows["parsed_answer_agreement"].mean()),
                "groups_n": int(len(group_rows)),
                "groups_of": EXPECTED_GROUPS,
            }
        )
    return model_sort(pd.DataFrame(rows))


def agreement_correctness(groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        group_rows = groups[groups["model_label"].eq(label)]
        if group_rows.empty:
            continue
        total = len(group_rows)
        rows.append(
            {
                "model_label": label,
                "model": display_name(label),
                "model_class": model_class(label),
                "same_correct_n": int(group_rows["same_correct"].sum()),
                "same_correct_pct": float(group_rows["same_correct"].mean()),
                "same_wrong_n": int(group_rows["same_wrong"].sum()),
                "same_wrong_pct": float(group_rows["same_wrong"].mean()),
                "disagreed_n": int(group_rows["disagreed"].sum()),
                "disagreed_pct": float(group_rows["disagreed"].mean()),
                "same_mixed_n": int(group_rows["same_mixed"].sum()),
                "same_mixed_pct": float(group_rows["same_mixed"].mean()),
                "groups": total,
            }
        )
    return model_sort(pd.DataFrame(rows))


def failure_counts(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        model_rows = scores[scores["model_label"].eq(label)]
        if model_rows.empty:
            continue
        rows.append(
            {
                "model_label": label,
                "model": display_name(label),
                "model_class": model_class(label),
                "empty_or_error_outputs": int((~model_rows["is_success"] | model_rows["is_empty_response"]).sum()),
                "status_errors": int((~model_rows["is_success"]).sum()),
                "empty_success_outputs": int((model_rows["is_success"] & model_rows["is_empty_response"]).sum()),
                "length_truncation_failures": int(model_rows["is_length_stop"].sum()),
                "parse_failures": int((model_rows["is_completed"] & ~model_rows["is_parsed"]).sum()),
                "responses": int(len(model_rows)),
            }
        )
    return model_sort(pd.DataFrame(rows))


def task_type_macro(scores: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    categories = [
        "Raw table",
        "Raw text",
        "Processed table",
        "Processed text",
        "Materials numeric",
        "Pairwise",
    ]
    per_model_rows: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        model_rows = scores[scores["model_label"].eq(label)]
        model_groups = groups[groups["model_label"].eq(label)]
        for category in categories:
            subset = model_rows[model_rows["task_category"].eq(category)]
            group_subset = model_groups[model_groups["task_category"].eq(category)]
            if subset.empty or group_subset.empty:
                continue
            per_model_rows.append(
                {
                    "model_label": label,
                    "task_category": category,
                    "response_n": int(len(subset)),
                    "group_n": int(len(group_subset)),
                    "repeat_accuracy": float(subset["is_correct_e2e"].mean()),
                    "parse_rate": float(subset["is_parsed"].mean()),
                    "exact_response_agreement": float(group_subset["exact_response_agreement"].mean()),
                    "parsed_answer_agreement": float(group_subset["parsed_answer_agreement"].mean()),
                }
            )
    per_model = pd.DataFrame(per_model_rows)
    rows: list[dict[str, Any]] = []
    for category in categories:
        subset = per_model[per_model["task_category"].eq(category)]
        if subset.empty:
            continue
        rows.append(
            {
                "task_category": category,
                "models": int(subset["model_label"].nunique()),
                "mean_responses_per_model": float(subset["response_n"].mean()),
                "mean_groups_per_model": float(subset["group_n"].mean()),
                "repeat_accuracy": float(subset["repeat_accuracy"].mean()),
                "parse_rate": float(subset["parse_rate"].mean()),
                "exact_response_agreement": float(subset["exact_response_agreement"].mean()),
                "parsed_answer_agreement": float(subset["parsed_answer_agreement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_summary(model: pd.DataFrame, agreement: pd.DataFrame) -> dict[str, Any]:
    top_acc = model.sort_values("overall_repeat_accuracy", ascending=False).iloc[0]
    max_agreement = model["exact_response_agreement"].max()
    top_agree = model[model["exact_response_agreement"].eq(max_agreement)]
    gap = model.copy()
    gap["agreement_accuracy_gap"] = gap["exact_response_agreement"] - gap["overall_repeat_accuracy"]
    repeatably_wrong = gap.sort_values(
        ["agreement_accuracy_gap", "exact_response_agreement"],
        ascending=[False, False],
    ).iloc[0]
    return {
        "models": int(len(model)),
        "expected_responses_per_model": EXPECTED_RESPONSES,
        "expected_groups_per_model": EXPECTED_GROUPS,
        "model_macro_repeat_accuracy": float(model["overall_repeat_accuracy"].mean()),
        "model_macro_parsed_only_accuracy": float(model["parsed_only_accuracy"].mean()),
        "model_macro_exact_response_agreement": float(model["exact_response_agreement"].mean()),
        "model_macro_parsed_answer_agreement": float(model["parsed_answer_agreement"].mean()),
        "highest_repeat_accuracy_model": str(top_acc["model"]),
        "highest_repeat_accuracy": float(top_acc["overall_repeat_accuracy"]),
        "highest_exact_agreement_models": top_agree["model"].tolist(),
        "highest_exact_agreement": float(max_agreement),
        "strongest_high_agreement_low_accuracy_model": str(repeatably_wrong["model"]),
        "strongest_high_agreement_low_accuracy_repeat_accuracy": float(
            repeatably_wrong["overall_repeat_accuracy"]
        ),
        "strongest_high_agreement_low_accuracy_exact_agreement": float(
            repeatably_wrong["exact_response_agreement"]
        ),
        "strongest_high_agreement_low_accuracy_gap": float(repeatably_wrong["agreement_accuracy_gap"]),
        "same_correct_macro": float(agreement["same_correct_pct"].mean()),
        "same_wrong_macro": float(agreement["same_wrong_pct"].mean()),
        "disagreed_macro": float(agreement["disagreed_pct"].mean()),
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_tex_model_summary(df: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_reliability_b3_analysis.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{\textbf{Repeated-prompt reliability by model.} Completed and parsed outputs are counted over 768 responses/model. Repeat accuracy counts all expected responses; parsed-only accuracy is conditional on successful parsing. Agreement is measured across 256 repeated prompt groups/model.}",
        r"\label{tab:reliability-b3-model-summary}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Completed & Parsed & \shortstack{Repeat\\acc.} & \shortstack{Parsed-only\\acc.} & \shortstack{Exact\\agree.} & \shortstack{Parsed\\agree.} \\",
        r"\midrule",
    ]
    for group_name, group in df.groupby("model_class", sort=False):
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{tex_escape(group_name)}}}}} \\")
        for row in group.itertuples(index=False):
            lines.append(
                f"{tex_escape(row.model)} & {row.completed_n}/768 & {row.parsed_n}/768 & "
                f"{pct(row.overall_repeat_accuracy)} & {pct(row.parsed_only_accuracy)} & "
                f"{pct(row.exact_response_agreement)} & {pct(row.parsed_answer_agreement)} \\\\"
            )
        lines.append(r"\addlinespace[1pt]")
    if lines[-1] == r"\addlinespace[1pt]":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "table_reliability_b3_model_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_tex_agreement(df: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_reliability_b3_analysis.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{\textbf{Agreement versus correctness in repeated prompts.} Counts and percentages are over 256 repeated prompt groups/model. Same-correct and same-wrong require all three responses to be exactly identical.}",
        r"\label{tab:reliability-b3-agreement-correctness}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & \shortstack{Same +\\correct} & \% & \shortstack{Same +\\wrong} & \% & Disagreed & \% \\",
        r"\midrule",
    ]
    for group_name, group in df.groupby("model_class", sort=False):
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{tex_escape(group_name)}}}}} \\")
        for row in group.itertuples(index=False):
            lines.append(
                f"{tex_escape(row.model)} & {row.same_correct_n} & {pct(row.same_correct_pct)} & "
                f"{row.same_wrong_n} & {pct(row.same_wrong_pct)} & "
                f"{row.disagreed_n} & {pct(row.disagreed_pct)} \\\\"
            )
        lines.append(r"\addlinespace[1pt]")
    if lines[-1] == r"\addlinespace[1pt]":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "table_reliability_b3_agreement_correctness.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_tex_failures(df: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_reliability_b3_analysis.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textbf{Repeated-prompt failure counts.} Empty/error outputs, length stops, and parsing failures are counted over 768 responses/model.}",
        r"\label{tab:reliability-b3-failures}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & \shortstack{Empty/\\error} & Length & Unparsed & Responses \\",
        r"\midrule",
    ]
    for group_name, group in df.groupby("model_class", sort=False):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{tex_escape(group_name)}}}}} \\")
        for row in group.itertuples(index=False):
            lines.append(
                f"{tex_escape(row.model)} & {row.empty_or_error_outputs} & "
                f"{row.length_truncation_failures} & {row.parse_failures} & {row.responses} \\\\"
            )
        lines.append(r"\addlinespace[1pt]")
    if lines[-1] == r"\addlinespace[1pt]":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "table_reliability_b3_failure_counts.tex").write_text("\n".join(lines), encoding="utf-8")


def write_tex_task_macro(df: pd.DataFrame) -> None:
    lines = [
        "% Candidate appendix table; generated by src/build_reliability_b3_analysis.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textbf{Repeated-prompt reliability by task type.} Values are model-macro percentages across the 13 included configurations.}",
        r"\label{tab:reliability-b3-task-type}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task type & \shortstack{Repeat\\acc.} & \shortstack{Parse\\rate} & \shortstack{Exact\\agree.} & \shortstack{Parsed\\agree.} \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{tex_escape(row.task_category)} & {pct(row.repeat_accuracy)} & {pct(row.parse_rate)} & "
            f"{pct(row.exact_response_agreement)} & {pct(row.parsed_answer_agreement)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "table_reliability_b3_task_type_macro.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    groups = group_records(scores)
    model = model_summary(scores, groups)
    agreement = agreement_correctness(groups)
    failures = failure_counts(scores)
    task_macro = task_type_macro(scores, groups)
    summary = build_summary(model, agreement)

    write_csv(OUTPUT_DIR / "reliability_b3_model_summary.csv", model)
    write_csv(OUTPUT_DIR / "reliability_b3_agreement_correctness.csv", agreement)
    write_csv(OUTPUT_DIR / "reliability_b3_failure_counts.csv", failures)
    write_csv(OUTPUT_DIR / "reliability_b3_task_type_macro.csv", task_macro)
    write_csv(OUTPUT_DIR / "reliability_b3_groups.csv", groups)
    write_json_atomic(OUTPUT_DIR / "reliability_b3_summary.json", summary)

    write_tex_model_summary(model)
    write_tex_agreement(agreement)
    write_tex_failures(failures)
    write_tex_task_macro(task_macro)

    print(f"Wrote B.3 reliability artifacts to {OUTPUT_DIR}")
    print(
        "Model-macro repeat accuracy: "
        f"{100.0 * summary['model_macro_repeat_accuracy']:.1f}%"
    )
    print(
        "Model-macro exact/parsed agreement: "
        f"{100.0 * summary['model_macro_exact_response_agreement']:.1f}% / "
        f"{100.0 * summary['model_macro_parsed_answer_agreement']:.1f}%"
    )


if __name__ == "__main__":
    main()
