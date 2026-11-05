#!/usr/bin/env python3
"""Build the comparative characterization error-analysis table."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ai4mat_common import REPO_ROOT


EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"
MANIFEST_PATH = REPO_ROOT / "results" / "manifests" / "case_study_image15.jsonl"
OUTPUT_DIR = REPO_ROOT / "paper" / "generated" / "potential"
EXCLUDED_MODELS = {"qwen3.6-27b", "bonsai-27b"}

OBJECTIVE_LABELS = {
    ("equivalent_diameter", "median"): "Larger median particle size",
    (
        "equivalent_diameter",
        "d90_minus_d10_over_d50",
    ): "Wider normalized size distribution",
    ("circularity", "median"): "More circular particle morphology",
}
OBJECTIVE_ORDER = [
    "Larger median particle size",
    "Wider normalized size distribution",
    "More circular particle morphology",
]

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
    "gemma-4-31b": "Gemma4 31B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "gpt-oss-20b-low": "GPT-OSS-20B (Low)",
    "gpt-oss-20b-medium": "GPT-OSS-20B (Medium)",
    "gpt-oss-20b": "GPT-OSS-20B (Default)",
    "gemma-4-26b-a4b": "Gemma4-26B-A4B",
    "bonsai-8b": "Bonsai-8B",
    "gpt-5.6-luna-medium": "GPT-5.6-Luna (Medium)",
    "gpt-5.6-luna-none": "GPT-5.6-Luna (Instant)",
    "gpt-5.4-mini-medium": "GPT-5.4-Mini (Medium)",
    "gpt-5.4-mini-none": "GPT-5.4-Mini (Instant)",
    "gpt-5.4-nano-medium": "GPT-5.4-Nano (Medium)",
    "gpt-5.4-nano-none": "GPT-5.4-Nano (Instant)",
}

MODEL_GROUPS = {
    "gemma-4-31b": "Open-weight / local",
    "qwen3.6-35b-a3b": "Open-weight / local",
    "gpt-oss-20b-low": "Open-weight / local",
    "gpt-oss-20b-medium": "Open-weight / local",
    "gpt-oss-20b": "Open-weight / local",
    "gemma-4-26b-a4b": "Open-weight / local",
    "bonsai-8b": "Open-weight / local",
    "gpt-5.6-luna-medium": "GPT series / API",
    "gpt-5.6-luna-none": "GPT series / API",
    "gpt-5.4-mini-medium": "GPT series / API",
    "gpt-5.4-mini-none": "GPT series / API",
    "gpt-5.4-nano-medium": "GPT series / API",
    "gpt-5.4-nano-none": "GPT series / API",
}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_case_scores() -> pd.DataFrame:
    candidates = [
        EVALUATION_DIR / "paper_case_study_scores.csv",
        EVALUATION_DIR / "paper_case_study_api_scores.csv",
        *sorted(EVALUATION_DIR.glob("paper_case_study_api_*_scores.csv")),
    ]
    frames = [pd.read_csv(path, low_memory=False) for path in candidates if path.exists()]
    if not frames:
        raise SystemExit("No paper case-study score files found.")
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[~scores["model_label"].astype(str).isin(EXCLUDED_MODELS)].copy()
    return scores.drop_duplicates(["model_label", "task_id"], keep="last")


def load_pairwise_margins() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("case_study_stage") != 1 or task.get("answer_type") != "label":
                continue
            metric_a = float(task["metric_a"])
            metric_b = float(task["metric_b"])
            denominator = (abs(metric_a) + abs(metric_b)) / 2.0
            if denominator == 0:
                symmetric_relative_difference = 0.0
            else:
                symmetric_relative_difference = abs(metric_a - metric_b) / denominator
            rows.append(
                {
                    "task_id": task["task_id"],
                    "objective": OBJECTIVE_LABELS.get(
                        (str(task.get("feature")), str(task.get("statistic"))),
                        f"{task.get('feature')} / {task.get('statistic')}",
                    ),
                    "symmetric_relative_difference": symmetric_relative_difference,
                }
            )
    return pd.DataFrame(rows)


def pairwise_scores(scores: pd.DataFrame | None = None) -> pd.DataFrame:
    if scores is None:
        scores = load_case_scores()
    pairwise = scores[
        scores["case_study_stage"].astype(str).eq("1")
        & scores["answer_type"].astype(str).eq("label")
    ].copy()
    pairwise = pairwise.merge(load_pairwise_margins(), on="task_id", how="left")
    pairwise["is_success"] = pairwise["status"].astype(str).eq("success")
    pairwise["is_empty_response"] = pairwise["response"].fillna("").astype(str).str.strip().eq("")
    pairwise["is_parsed"] = pairwise["is_success"] & ~pairwise["is_empty_response"] & truthy(pairwise["parse_success"])
    pairwise["is_correct_e2e"] = pairwise["is_success"] & truthy(pairwise["parse_success"]) & truthy(pairwise["correct"])
    return pairwise


def model_summary(scores: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    topk = scores[
        scores["case_study_stage"].astype(str).eq("2")
        & scores["answer_type"].astype(str).eq("topk_metrics")
    ].copy()
    rows: list[dict[str, Any]] = []
    for model_label in MODEL_ORDER:
        model_pairwise = pairwise[pairwise["model_label"].astype(str).eq(model_label)]
        model_topk = topk[topk["model_label"].astype(str).eq(model_label)]
        raw = model_pairwise[model_pairwise["encoding"].astype(str).eq("raw_text")]
        processed = model_pairwise[model_pairwise["encoding"].astype(str).eq("routed_json")]
        raw_accuracy = float(raw["is_correct_e2e"].mean()) if len(raw) else float("nan")
        processed_accuracy = (
            float(processed["is_correct_e2e"].mean()) if len(processed) else float("nan")
        )
        exact_top3 = (
            pd.to_numeric(model_topk["topk_precision_at_3"], errors="coerce").fillna(0).mean()
            if len(model_topk)
            else float("nan")
        )
        family_top3 = (
            pd.to_numeric(model_topk["topk_family_precision_at_3"], errors="coerce")
            .fillna(0)
            .mean()
            if len(model_topk)
            else float("nan")
        )
        rows.append(
            {
                "group": MODEL_GROUPS[model_label],
                "model_label": model_label,
                "model": MODEL_NAMES[model_label],
                "pairwise_raw_accuracy": raw_accuracy,
                "pairwise_processed_accuracy": processed_accuracy,
                "improvement_pp": 100.0 * (processed_accuracy - raw_accuracy),
                "exact_top3_p_at_3": exact_top3,
                "family_top3_p_at_3": family_top3,
                "pairwise_raw_n": int(len(raw)),
                "pairwise_processed_n": int(len(processed)),
                "top3_n": int(len(model_topk)),
            }
        )
    return pd.DataFrame(rows)


def summarize(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for objective in OBJECTIVE_ORDER:
        subset = pairwise[pairwise["objective"].eq(objective)]
        rows.append(summary_row(objective, subset, bold=False))
    rows.append(summary_row("Overall", pairwise, bold=True))
    return pd.DataFrame(rows)


def summary_row(objective: str, subset: pd.DataFrame, *, bold: bool) -> dict[str, Any]:
    raw = subset[subset["encoding"].astype(str).eq("raw_text")]
    processed = subset[subset["encoding"].astype(str).eq("routed_json")]
    raw_accuracy = float(raw["is_correct_e2e"].mean()) if len(raw) else float("nan")
    processed_accuracy = float(processed["is_correct_e2e"].mean()) if len(processed) else float("nan")
    correct = subset[subset["is_correct_e2e"]]
    failed = subset[~subset["is_correct_e2e"]]
    return {
        "characterization_objective": objective,
        "raw_n": int(len(raw)),
        "raw_parsed_n": int(raw["is_parsed"].sum()),
        "raw_correct_n": int(raw["is_correct_e2e"].sum()),
        "raw_particle_records": raw_accuracy,
        "processed_n": int(len(processed)),
        "processed_parsed_n": int(processed["is_parsed"].sum()),
        "processed_correct_n": int(processed["is_correct_e2e"].sum()),
        "processed_values": processed_accuracy,
        "improvement_pp": 100.0 * (processed_accuracy - raw_accuracy),
        "median_difference_correct": float(correct["symmetric_relative_difference"].median())
        if len(correct)
        else float("nan"),
        "median_difference_failed": float(failed["symmetric_relative_difference"].median())
        if len(failed)
        else float("nan"),
        "failed_n": int(len(failed)),
        "bold": bold,
    }


def numeric_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{100.0 * number:.1f}\\%"


def pp(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.1f} pp"


def maybe_bold(text: str, bold: bool) -> str:
    return rf"\textbf{{{text}}}" if bold else text


def objective_tex(label: str, bold: bool) -> str:
    if label == "Overall":
        return r"\textbf{Overall}"
    return label


def write_latex(summary: pd.DataFrame) -> None:
    lines = [
        "% Generated by src/build_comparative_characterization_error_table.py",
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{\textbf{Comparative characterization performance by input condition and objective.}",
        r"Pairwise accuracy across raw and processed inputs, with median differences for correct and failed comparisons.}",
        r"\label{tab:comparative-characterization-error}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Accuracy} & \multicolumn{2}{c}{Median difference} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
        r"Objective & Raw & Processed & $\Delta$ & Correct & Failed \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        bold = bool(row.bold)
        if row.characterization_objective == "Overall":
            lines.append(r"\midrule")
        lines.append(
            f"{objective_tex(row.characterization_objective, bold)}\n"
            f"& "
            f"{maybe_bold(numeric_pct(row.raw_particle_records), bold)} & "
            f"{maybe_bold(numeric_pct(row.processed_values), bold)} & "
            f"{maybe_bold(pp(row.improvement_pp), bold)} & "
            f"{maybe_bold(numeric_pct(row.median_difference_correct), bold)} & "
            f"{maybe_bold(numeric_pct(row.median_difference_failed), bold)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "table_comparative_characterization_error_analysis.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def validate_model_results(model_results: pd.DataFrame) -> dict[str, Any]:
    missing = [
        model_label
        for model_label in MODEL_ORDER
        if model_label not in set(model_results["model_label"].astype(str))
    ]
    if missing:
        raise SystemExit(f"Missing model rows: {', '.join(missing)}")

    bad_counts = model_results[
        ~(
            model_results["pairwise_raw_n"].eq(90)
            & model_results["pairwise_processed_n"].eq(90)
            & model_results["top3_n"].eq(30)
        )
    ]
    if len(bad_counts):
        cols = ["model_label", "pairwise_raw_n", "pairwise_processed_n", "top3_n"]
        raise SystemExit(f"Unexpected source counts:\n{bad_counts[cols].to_string(index=False)}")

    macro = {
        "raw": 100.0 * float(model_results["pairwise_raw_accuracy"].mean()),
        "processed": 100.0 * float(model_results["pairwise_processed_accuracy"].mean()),
        "delta": float(model_results["improvement_pp"].mean()),
        "exact_top3": 100.0 * float(model_results["exact_top3_p_at_3"].mean()),
        "family_top3": 100.0 * float(model_results["family_top3_p_at_3"].mean()),
    }
    return {
        "model_configurations": int(len(model_results)),
        "macro": macro,
        "min_improvement_pp": float(model_results["improvement_pp"].min()),
        "max_improvement_pp": float(model_results["improvement_pp"].max()),
        "processed_ge_97_8": int((100.0 * model_results["pairwise_processed_accuracy"] >= 97.8).sum()),
    }


def write_model_latex(model_results: pd.DataFrame, checks: dict[str, Any]) -> None:
    del checks  # The validation summary is printed for review; the table remains compact.
    lines = [
        "% Generated by src/build_comparative_characterization_error_table.py",
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{\textbf{Comparative characterization performance across model configurations.}",
        r"Pairwise accuracy across raw and processed inputs and performance on open-ended top-3 morphology selection.}",
        r"\label{tab:comparative-characterization-models}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Pairwise accuracy} & \multicolumn{2}{c}{Open-ended top-3} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
        r"Model & Raw & Processed & $\Delta$ & Exact P\@3 & Family P\@3 \\",
        r"\midrule",
    ]

    for group in ["Open-weight / local", "GPT series / API"]:
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{group}}}}} \\")
        for row in model_results[model_results["group"].astype(str).eq(group)].itertuples(index=False):
            lines.append(
                f"{row.model} & "
                f"{numeric_pct(row.pairwise_raw_accuracy)} & "
                f"{numeric_pct(row.pairwise_processed_accuracy)} & "
                f"{pp(row.improvement_pp)} & "
                f"{numeric_pct(row.exact_top3_p_at_3)} & "
                f"{numeric_pct(row.family_top3_p_at_3)} \\\\"
            )
        if group == "Open-weight / local":
            lines.append(r"\midrule")

    average_cells = [
        numeric_pct(model_results["pairwise_raw_accuracy"].mean()),
        numeric_pct(model_results["pairwise_processed_accuracy"].mean()),
        pp(model_results["improvement_pp"].mean()),
        numeric_pct(model_results["exact_top3_p_at_3"].mean()),
        numeric_pct(model_results["family_top3_p_at_3"].mean()),
    ]
    lines.extend(
        [
            r"\midrule",
            r"\textbf{Average} & "
            + " & ".join(rf"\textbf{{{cell}}}" for cell in average_cells)
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (OUTPUT_DIR / "table_comparative_characterization_models.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_case_scores()
    pairwise = pairwise_scores(scores)
    summary = summarize(pairwise)
    model_results = model_summary(scores, pairwise)
    summary.drop(columns=["bold"]).to_csv(
        OUTPUT_DIR / "comparative_characterization_error_analysis.csv",
        index=False,
    )
    combo_rows = []
    for objective in OBJECTIVE_ORDER:
        subset = pairwise[pairwise["objective"].eq(objective)]
        for encoding, input_condition in [
            ("raw_text", "Raw particles"),
            ("routed_json", "Processed"),
        ]:
            rows = subset[subset["encoding"].astype(str).eq(encoding)]
            combo_rows.append(
                {
                    "characterization_objective": objective,
                    "input_condition": input_condition,
                    "n": int(len(rows)),
                    "parsed_n": int(rows["is_parsed"].sum()),
                    "correct_n": int(rows["is_correct_e2e"].sum()),
                    "accuracy": float(rows["is_correct_e2e"].mean()) if len(rows) else float("nan"),
                }
            )
    pd.DataFrame(combo_rows).to_csv(
        OUTPUT_DIR / "comparative_pairwise_by_condition_objective.csv",
        index=False,
    )
    model_results.to_csv(
        OUTPUT_DIR / "comparative_characterization_by_model.csv",
        index=False,
    )
    checks = validate_model_results(model_results)
    write_latex(summary)
    write_model_latex(model_results, checks)
    print(f"Wrote {OUTPUT_DIR / 'comparative_characterization_error_analysis.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'comparative_pairwise_by_condition_objective.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'comparative_characterization_by_model.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'table_comparative_characterization_error_analysis.tex'}")
    print(f"Wrote {OUTPUT_DIR / 'table_comparative_characterization_models.tex'}")
    print(
        "Verification: "
        f"{checks['model_configurations']} model configurations; "
        "counts are 90 raw pairwise, 90 processed pairwise, and 30 top-3 prompts/model."
    )
    macro = checks["macro"]
    print(
        "Macro averages: "
        f"Raw {macro['raw']:.1f}%, "
        f"Processed {macro['processed']:.1f}%, "
        f"Delta {macro['delta']:+.1f} pp, "
        f"Exact P@3 {macro['exact_top3']:.1f}%, "
        f"Family P@3 {macro['family_top3']:.1f}%."
    )
    print(
        "Improvement range: "
        f"{checks['min_improvement_pp']:+.1f} to {checks['max_improvement_pp']:+.1f} pp; "
        f"{checks['processed_ge_97_8']} configurations reached at least 97.8% processed pairwise accuracy."
    )


if __name__ == "__main__":
    main()
