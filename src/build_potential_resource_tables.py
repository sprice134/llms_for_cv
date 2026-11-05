#!/usr/bin/env python3
"""Build Table 1-style potential tables for token use and latency.

These tables are intentionally outside the main paper artifact builder. They
reuse the model order and task groupings from the potential analysis sandbox,
but report resource burden instead of accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_potential_analyses import (
    EXCLUDED_MODELS,
    MODEL_NAMES,
    MODEL_ORDER,
    OUTPUT_DIR,
    load_base,
    load_case,
    load_reliability,
    load_tool,
)


API_MODELS = {
    "gpt-5.6-luna-medium",
    "gpt-5.6-luna-none",
    "gpt-5.4-mini-medium",
    "gpt-5.4-mini-none",
    "gpt-5.4-nano-medium",
    "gpt-5.4-nano-none",
}

TABLE_COLUMNS = [
    ("raw_table", "Raw", "table"),
    ("raw_text", "Raw", "text"),
    ("processed_table", "Proc.", "table"),
    ("processed_text", "Proc.", "text"),
    ("tool_call", "Tool", "call"),
    ("repeat_test", "Repeat", "test"),
    ("case_pairwise", "Pairwise", "comp."),
    ("case_family_top3", "Family", "top-3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-field",
        choices=("total_tokens", "input_tokens", "output_tokens"),
        default=None,
        help=(
            "Deprecated compatibility option. The script now writes total, input, "
            "and output token tables in one run."
        ),
    )
    return parser.parse_args()


def model_name(label: str) -> str:
    return MODEL_NAMES.get(label, label)


def model_class(label: str) -> str:
    return "API" if label in API_MODELS else "Open"


def ordered_model_labels(*frames: pd.DataFrame) -> list[str]:
    present: set[str] = set()
    for frame in frames:
        if frame.empty or "model_label" not in frame:
            continue
        present.update(frame["model_label"].dropna().astype(str))
    present -= EXCLUDED_MODELS
    order = {label: index for index, label in enumerate(MODEL_ORDER)}
    return sorted(present, key=lambda label: (order.get(label, len(order)), label))


def numeric_mean(rows: pd.DataFrame, column: str) -> float:
    if rows.empty or column not in rows:
        return float("nan")
    values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else float("nan")


def direct_value(base: pd.DataFrame, label: str, encoding: str, metric_column: str) -> float:
    rows = base[
        base["model_label"].astype(str).eq(label)
        & base["suite"].astype(str).eq("core")
        & base["encoding"].astype(str).eq(encoding)
    ]
    return numeric_mean(rows, metric_column)


def case_value(case: pd.DataFrame, label: str, column_key: str, metric_column: str) -> float:
    rows = case[case["model_label"].astype(str).eq(label)]
    if rows.empty:
        return float("nan")
    if column_key == "case_pairwise":
        rows = rows[
            rows["case_study_stage"].astype(str).eq("1")
            & rows["answer_type"].astype(str).eq("label")
        ]
    elif column_key == "case_family_top3":
        rows = rows[
            rows["case_study_stage"].astype(str).eq("2")
            & rows["answer_type"].astype(str).eq("topk_metrics")
        ]
    return numeric_mean(rows, metric_column)


def collect_metric_table(metric_column: str) -> pd.DataFrame:
    base = load_base()
    tool = load_tool()
    case = load_case()
    reliability = load_reliability()
    labels = ordered_model_labels(base, tool, case, reliability)

    rows: list[dict[str, Any]] = []
    for label in labels:
        tool_rows = tool[tool["model_label"].astype(str).eq(label)]
        repeat_rows = reliability[reliability["model_label"].astype(str).eq(label)]
        row: dict[str, Any] = {
            "model_label": label,
            "model": model_name(label),
            "model_class": model_class(label),
            "raw_table": direct_value(base, label, "raw_table", metric_column),
            "raw_text": direct_value(base, label, "raw_text", metric_column),
            "processed_table": direct_value(base, label, "processed_table", metric_column),
            "processed_text": direct_value(base, label, "processed_text", metric_column),
            "tool_call": numeric_mean(tool_rows, metric_column),
            "repeat_test": numeric_mean(repeat_rows, metric_column),
            "case_pairwise": case_value(case, label, "case_pairwise", metric_column),
            "case_family_top3": case_value(case, label, "case_family_top3", metric_column),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def tex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def format_token(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:,.0f}"


def format_latency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.2f}"


def table_rows(frame: pd.DataFrame, value_formatter: Any) -> tuple[list[list[str]], list[list[str]]]:
    open_rows: list[list[str]] = []
    api_rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rendered = [tex_escape(row["model"])]
        for key, _, _ in TABLE_COLUMNS:
            rendered.append(value_formatter(row.get(key)))
        if row["model_class"] == "API":
            api_rows.append(rendered)
        else:
            open_rows.append(rendered)
    return open_rows, api_rows


def write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    value_formatter: Any,
) -> None:
    open_rows, api_rows = table_rows(frame, value_formatter)
    lines = [
        "% Auto-generated by src/build_potential_resource_tables.py",
        "\\begin{table}[!h]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\newcommand{\\resdirecthead}[2]{\\parbox[c][2.45em][c]{0.68cm}{\\centering #1\\\\#2}}",
        "\\newcommand{\\resmetrichead}[2]{\\parbox[c][2.45em][c]{1.05cm}{\\centering #1\\\\#2}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Direct answer} & \\multicolumn{1}{c}{Tool call} & \\multicolumn{1}{c}{Repeat test} & \\multicolumn{2}{c}{Case study} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-7}\\cmidrule(l){8-9}",
        "Model"
        + " & "
        + " & ".join(
            [
                "\\resdirecthead{Raw}{table}",
                "\\resdirecthead{Raw}{text}",
                "\\resdirecthead{Proc.}{table}",
                "\\resdirecthead{Proc.}{text}",
                "\\resmetrichead{All}{calls}",
                "\\resmetrichead{Repeat}{prompts}",
                "\\resmetrichead{Pairwise}{comp.}",
                "\\resmetrichead{Family}{top-3}",
            ]
        )
        + " \\\\",
        "\\midrule",
    ]
    groups = [
        ("Open-weight / local", open_rows),
        ("GPT series / API", api_rows),
    ]
    wrote_group = False
    for group_name, rows in groups:
        if not rows:
            continue
        if wrote_group:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{tex_escape(group_name)}}}}} \\\\")
        for row in rows:
            lines.append(" & ".join(row) + " \\\\")
        wrote_group = True
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def write_readme() -> None:
    text = """# Potential Resource Tables

This folder includes Table 1-style exploratory resource tables:

- `table_token_cost_table1_style.tex`: mean total tokens per completed response by model and task family.
- `table_input_token_cost_table1_style.tex`: mean input tokens per completed response by model and task family.
- `table_output_token_cost_table1_style.tex`: mean output tokens per completed response by model and task family.
- `table_latency_table1_style.tex`: mean wall-clock seconds per completed response by model and task family.

The direct-answer columns match the main Table 1 input-format layout. The tool-call section is collapsed to one column because accepted and strict scoring are alternative evaluations of the same generated response, so their token and latency costs are identical at inference time.
"""
    (OUTPUT_DIR / "resource_tables_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.token_field is not None:
        print(
            "--token-field is ignored; writing total, input, and output token "
            "tables in one run."
        )

    token_tables = {
        "total_tokens": {
            "csv": "token_cost_table1_style.csv",
            "tex": "table_token_cost_table1_style.tex",
            "label": "tab:potential-token-cost",
            "name": "total tokens",
        },
        "input_tokens": {
            "csv": "input_token_cost_table1_style.csv",
            "tex": "table_input_token_cost_table1_style.tex",
            "label": "tab:potential-input-token-cost",
            "name": "input tokens",
        },
        "output_tokens": {
            "csv": "output_token_cost_table1_style.csv",
            "tex": "table_output_token_cost_table1_style.tex",
            "label": "tab:potential-output-token-cost",
            "name": "output tokens",
        },
    }
    latency_table = collect_metric_table("elapsed_seconds")

    for metric_column, spec in token_tables.items():
        token_table = collect_metric_table(metric_column)
        token_table.to_csv(OUTPUT_DIR / str(spec["csv"]), index=False)
        token_caption = (
            f"\\textbf{{Mean {spec['name']} by evaluation condition.}} Values are "
            f"mean {spec['name']} per completed response, grouped with the same "
            "model order and task families as Table~\\ref{tab:main-results}."
        )
        write_latex_table(
            OUTPUT_DIR / str(spec["tex"]),
            token_table,
            caption=token_caption,
            label=str(spec["label"]),
            value_formatter=format_token,
        )

    latency_table.to_csv(OUTPUT_DIR / "latency_table1_style.csv", index=False)

    latency_caption = (
        "\\textbf{Mean latency by evaluation condition.} Values are mean wall-clock "
        "seconds per completed response, grouped with the same model order and task "
        "families as Table~\\ref{tab:main-results}."
    )
    write_latex_table(
        OUTPUT_DIR / "table_latency_table1_style.tex",
        latency_table,
        caption=latency_caption,
        label="tab:potential-latency",
        value_formatter=format_latency,
    )
    write_readme()
    print(f"Wrote resource tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
