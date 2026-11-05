#!/usr/bin/env python3
"""Audit released task coverage and the paper's headline results."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'results/evaluation'
MANIFEST = ROOT / 'results/manifests'
SUMMARY = ROOT / 'paper/generated/potential'


def csv_rows(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        yield from csv.DictReader(handle)


def manifest_ids(name: str, expected: int) -> set[str]:
    path = MANIFEST / (name + '.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        ids = {json.loads(line)['task_id'] for line in handle if line.strip()}
    assert len(ids) == expected, (name, len(ids), expected)
    return ids


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def success(row: dict[str, str]) -> bool:
    return row['status'] == 'success' and row['parse_success'] == 'True'


def correct(row: dict[str, str]) -> bool:
    return success(row) and row['correct'] == 'True'


def check_pct(name: str, value: float, expected: float) -> None:
    assert round(value * 100, 1) == expected, (name, value * 100, expected)
    print(f'{name}: {value * 100:.1f}%')


def main() -> None:
    manifests = {
        'base': manifest_ids('tasks_image20.jsonl', 5500),
        'tool': manifest_ids('tool_routing_image15.jsonl', 555),
        'repeat': manifest_ids('reliability_image15_repeat3.jsonl', 768),
        'case': manifest_ids('case_study_image15.jsonl', 360),
    }
    files = {
        'base': [EVAL / 'paper_base_auto_scores.csv.gz'],
        'tool': sorted(EVAL.glob('paper_tool_*_scores.csv.gz')),
        'repeat': sorted(EVAL.glob('paper_reliability_*_scores.csv.gz')),
        'case': [EVAL / 'paper_case_study_scores.csv.gz']
        + sorted(EVAL.glob('paper_case_study_api_*_scores.csv.gz')),
    }
    rows = {suite: [row for path in paths for row in csv_rows(path)]
            for suite, paths in files.items()}
    expected_per_model = {'base': 5500, 'tool': 555, 'repeat': 768, 'case': 360}
    all_models = {row['model_label'] for row in rows['base']}
    assert len(all_models) == 13
    for suite, suite_rows in rows.items():
        by_model: dict[str, set[str]] = defaultdict(set)
        for row in suite_rows:
            model, task_id = row['model_label'], row['task_id']
            assert model in all_models, (suite, model)
            assert task_id in manifests[suite], (suite, task_id)
            assert task_id not in by_model[model], (suite, model, task_id)
            by_model[model].add(task_id)
        assert set(by_model) == all_models, suite
        assert all(len(ids) == expected_per_model[suite] for ids in by_model.values()), suite
        print(f'{suite}: {len(suite_rows):,} scored responses')
    assert sum(map(len, rows.values())) == 93_379

    base = rows['base']
    core = [r for r in base if r['suite'] == 'core']
    assert len(core) == 13 * 5120
    def rate_for(items) -> float:
        items = list(items)
        return mean(correct(r) for r in items)
    raw = rate_for(r for r in core if r['encoding'].startswith('raw_'))
    processed = rate_for(r for r in core if r['encoding'].startswith('processed_'))
    check_pct('Direct raw', raw, 48.8)
    check_pct('Direct processed', processed, 92.0)
    check_pct('Direct paired gap', processed - raw, 43.2)
    check_pct('Materials numeric raw', rate_for(r for r in base if r['suite'] == 'interface_numeric' and r['encoding'] == 'raw_text'), 46.8)
    check_pct('Materials numeric processed', rate_for(r for r in base if r['suite'] == 'interface_numeric' and r['encoding'] == 'routed_json'), 100.0)
    check_pct('Materials pairwise raw', rate_for(r for r in base if r['suite'] == 'interface_pairwise' and r['encoding'] == 'raw_text'), 65.6)
    check_pct('Materials pairwise processed', rate_for(r for r in base if r['suite'] == 'interface_pairwise' and r['encoding'] == 'routed_json'), 78.4)

    tool = rows['tool']
    accepted = mean(r['status'] == 'success' and r['tool_routing_correct'] == 'True' for r in tool)
    strict = mean(r['status'] == 'success' and r['tool_routing_strict_correct'] == 'True' for r in tool)
    check_pct('Tool accepted', accepted, 89.7)
    check_pct('Tool strict', strict, 80.7)
    routing_by_model = Counter()
    for r in tool:
        routing_by_model[r['model_label']] += r['status'] == 'success' and r['tool_routing_correct'] == 'True'
    assert sum(n / 555 >= 0.95 for n in routing_by_model.values()) == 10
    print('Accepted routing >=95%: 10/13 configurations')

    check_pct('Repeat accuracy', rate_for(rows['repeat']), 70.3)
    case = rows['case']
    pairwise = [r for r in case if r['suite'] == 'case_study_stage1_pairwise']
    check_pct('Comparative pairwise raw', rate_for(r for r in pairwise if r['encoding'] == 'raw_text'), 72.5)
    check_pct('Comparative pairwise processed', rate_for(r for r in pairwise if r['encoding'] == 'routed_json'), 94.3)
    top3 = [r for r in case if r['suite'] == 'case_study_stage2_topk']
    for column, label, expected in (
        ('topk_precision_at_3', 'Top-3 exact precision', 5.6),
        ('topk_family_precision_at_3', 'Top-3 family precision', 35.9),
    ):
        value = mean(float(r[column]) if r[column] else 0.0 for r in top3)
        check_pct(label, value, expected)

    ci = next(csv_rows(SUMMARY / 'claim_confidence_intervals.csv'))
    assert ci['comparison'] == 'Processed direct - raw direct'
    assert abs(float(ci['first_mean']) - processed) < 1e-12
    assert abs(float(ci['second_mean']) - raw) < 1e-12
    check_pct('Direct gap CI lower', float(ci['difference_ci95_low']), 36.1)
    check_pct('Direct gap CI upper', float(ci['difference_ci95_high']), 52.6)
    print('PASS: 93,379 scored responses, 13 models, and headline results')


if __name__ == '__main__':
    main()
